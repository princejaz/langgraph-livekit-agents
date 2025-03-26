import { llm, tts } from "@livekit/agents";
import {
  AIMessageChunk,
  BaseMessageChunk,
  BaseMessageLike,
  MessageContent,
  MessageContentComplex,
  MessageType,
} from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { pipeline } from "@livekit/agents";
import {
  Command,
  CompiledGraph,
  LangGraphRunnableConfig,
  MemorySaver,
} from "@langchain/langgraph";

type VoiceEvent = { type: "say"; data: { source: string } } | { type: "flush" };

const saver = new MemorySaver();

type AnyCompiledGraph = CompiledGraph<any, any, any, any, any, any>;

class LangGraphStream extends llm.LLMStream {
  label = "langgraph.LangGraphStream";

  #graph: AnyCompiledGraph;
  #controller: AbortController;
  #configurable: LangGraphRunnableConfig["configurable"];

  constructor(
    llm: llm.LLM,
    graph: AnyCompiledGraph,
    chatCtx: llm.ChatContext,
    fncCtx: llm.FunctionContext,
    options?: {
      flush?: () => void;
      configurable: LangGraphRunnableConfig["configurable"];
    }
  ) {
    super(llm, chatCtx, {});
    this.#graph = graph;

    this.#controller = new AbortController();
    this.#configurable = options?.configurable;

    // set the checkpointer to the saver, but now we need to set the thread_id everywhere
    this.#graph.checkpointer ??= saver;

    this.#run();
  }

  async #getInterrupt() {
    const state = await this.#graph.getState({
      configurable: this.#configurable,
    });

    const interrupts = state.tasks.flatMap((task) => task.interrupts);
    const [interrupt] = interrupts
      .reverse()
      .filter((interrupt) => typeof interrupt.value === "string");

    return interrupt;
  }

  async #run() {
    // push to queue to prevent ttft being undefined when sending metrics
    this.queue.put({ requestId: "<unknown>", choices: [] });

    const messages = await Promise.all(this.chatCtx.messages.map(toMessage));
    const interrupt = await this.#getInterrupt();

    // check if we're interrupted, if so, send continue instead
    let input: { messages: BaseMessageLike | BaseMessageLike[] } | Command = {
      messages,
    };

    if (interrupt) {
      const lastMessage = messages.at(-1);
      if (lastMessage != null) {
        input = new Command({
          resume: {
            content: lastMessage.content,
            messages: [
              { type: "ai", content: interrupt.value },
              { type: "human", content: lastMessage.content },
            ],
          },
        });
      }
    }

    try {
      for await (const payload of await this.#graph.stream(input, {
        signal: this.#controller.signal,
        configurable: this.#configurable,
        streamMode: ["messages", "custom"],
      })) {
        const [event, data] = payload;
        if (event === "messages") {
          const [message] = data as [BaseMessageChunk, RunnableConfig];
          const chunk = await toLkChunk(message);
          if (chunk) this.queue.put(chunk);
        } else if (event === "custom") {
          const event = data as VoiceEvent;
          if (event.type === "say") {
            const lkChunk = await toLkChunk(
              new AIMessageChunk({ content: event.data.source })
            );
            if (lkChunk) this.queue.put(lkChunk);
          } else if (event.type === "flush") {
            // @ts-expect-error Flushing sentinel
            this.queue.put(tts.SynthesizeStream.FLUSH_SENTINEL);
          }
        }
      }

      // check the state again if we were interrupted
      const interrupt = await this.#getInterrupt();
      if (interrupt) {
        const lkChunk = await toLkChunk(
          new AIMessageChunk({ content: interrupt.value })
        );
        if (lkChunk) this.queue.put(lkChunk);
      }
    } finally {
      this.queue.close();
    }
  }

  abort() {
    this.#controller.abort();
  }
}

function isLkChatImage(i: llm.ChatContent): i is llm.ChatImage {
  return typeof i === "object" && "image" in i && i.image != null;
}

async function toMessage(m: llm.ChatMessage): Promise<{
  id: string | undefined;
  type: MessageType;
  content: MessageContent;
}> {
  const type: MessageType =
    {
      [llm.ChatRole.USER]: "human" as const,
      [llm.ChatRole.ASSISTANT]: "ai" as const,
      [llm.ChatRole.SYSTEM]: "system" as const,
      [llm.ChatRole.TOOL]: "tool" as const,
    }[m.role] ?? ("human" as const);

  let content: MessageContent = "";

  if (typeof m.content === "string") {
    content = m.content;
  } else if (Array.isArray(m.content)) {
    content = await Promise.all(
      m.content.map((c): MessageContentComplex => {
        if (typeof c === "string") return { type: "text", text: c };

        if (isLkChatImage(c)) {
          if (typeof c.image === "string") {
            return { type: "image_url", image_url: c.image };
          }

          throw new Error("Unsupported image type");
        }

        throw new Error("Unsupported content type");
      })
    );
  }

  return { id: m.id, type, content };
}

async function toLkChunk(m: BaseMessageChunk): Promise<llm.ChatChunk | null> {
  if (m.content === undefined || typeof m.content !== "string") return null;
  return {
    requestId: m.id ?? "<unknown>",
    choices: [
      { delta: { content: m.content, role: llm.ChatRole.ASSISTANT }, index: 0 },
    ],
  };
}

export class LangGraph extends llm.LLM {
  #graph: AnyCompiledGraph;
  #config: Pick<LangGraphRunnableConfig, "configurable">;
  constructor(
    graph: AnyCompiledGraph,
    config: Pick<LangGraphRunnableConfig, "configurable">
  ) {
    super();
    this.#graph = graph;
    this.#config = config;
  }

  chat(params: {
    chatCtx: llm.ChatContext;
    fncCtx: llm.FunctionContext;
    signal: AbortSignal;
    say?: (
      source: string,
      allowInterruptions: boolean,
      addToChatCtx: boolean
    ) => void;
  }): llm.LLMStream {
    const { chatCtx, fncCtx, ...options } = params;
    return new LangGraphStream(this, this.#graph, chatCtx, fncCtx, {
      configurable: this.#config.configurable,
      ...options,
    });
  }
}

type VAD = ConstructorParameters<typeof pipeline.VoicePipelineAgent>[0];
type STT = ConstructorParameters<typeof pipeline.VoicePipelineAgent>[1];
type TTS = ConstructorParameters<typeof pipeline.VoicePipelineAgent>[3];

type VoicePipelineAgentOptions = ConstructorParameters<
  typeof pipeline.VoicePipelineAgent
>[4] &
  Pick<LangGraphRunnableConfig, "configurable">;

export class LangGraphAgent extends pipeline.VoicePipelineAgent {
  constructor(
    vad: VAD,
    stt: STT,
    graph: AnyCompiledGraph,
    tts: TTS,
    opts: VoicePipelineAgentOptions
  ) {
    const llm = new LangGraph(graph, opts);
    const beforeLLMCallback: pipeline.BeforeLLMCallback = (agent, chatCtx) => {
      const signal = new AbortController();
      agent.once(pipeline.VPAEvent.AGENT_SPEECH_INTERRUPTED, () =>
        signal.abort()
      );

      return (agent.llm as LangGraph).chat({
        chatCtx,
        fncCtx: agent.fncCtx ?? {},
        signal: signal.signal,
        say: agent.say.bind(agent),
      });
    };

    super(vad, stt, llm, tts, { ...opts, beforeLLMCallback });
  }
}
