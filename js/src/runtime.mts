import { llm } from "@livekit/agents";
import {
  AIMessageChunk,
  BaseMessageChunk,
  type MessageContent,
  type MessageContentComplex,
  type MessageType,
} from "@langchain/core/messages";
import {
  Command,
  isGraphInterrupt,
  type LangGraphRunnableConfig,
  type Pregel,
} from "@langchain/langgraph";

type VoiceEvent = { type: "say"; data: { source: string } };
type AnyPregelInterface = Pick<
  Pregel<any, any, any>,
  "invoke" | "stream" | "getState"
>;

interface LangGraphOptions {
  configurable?: LangGraphRunnableConfig["configurable"];
  messageKey?: string;
}

class LangGraphStream extends llm.LLMStream {
  label = "langgraph.LangGraphStream";

  #graph: AnyPregelInterface;
  #options: LangGraphOptions;

  constructor(
    llm: llm.LLM,
    graph: AnyPregelInterface,
    chatCtx: llm.ChatContext,
    fncCtx: llm.FunctionContext,
    options: LangGraphOptions
  ) {
    super(llm, chatCtx, {});
    this.#graph = graph;
    this.#options = options;

    this.#run();
  }

  async #getInterrupt() {
    if (!this.#options?.configurable) return undefined;
    try {
      const state = await this.#graph.getState({
        configurable: this.#options?.configurable,
      });

      const interrupts = state.tasks.flatMap((task) => task.interrupts);
      const [interrupt] = interrupts
        .reverse()
        .filter((interrupt) => typeof interrupt.value === "string");

      return interrupt;
    } catch {
      return undefined;
    }
  }

  async #run() {
    // push to queue to prevent ttft being undefined when sending metrics
    this.queue.put({ requestId: "<unknown>", choices: [] });

    const messages = await Promise.all(
      this.chatCtx.messages.map(toLangChainMessage)
    );
    const interrupt = await this.#getInterrupt();

    // check if we're interrupted, if so, send continue instead
    let input: unknown | Command = {
      [this.#options?.messageKey ?? "messages"]: messages,
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
      // We need to wrap it in a second try/catch
      // in order to serve both RemoteGraph and StateGraph
      // RemoteGraph will throw an error if interrupted,
      // whereas StateGraph will not.
      try {
        for await (const payload of await this.#graph.stream(input, {
          configurable: this.#options?.configurable,
          streamMode: ["messages", "custom"],
        })) {
          const [event, data] = payload;
          if (event === "messages") {
            const [message] = data as [
              BaseMessageChunk,
              LangGraphRunnableConfig
            ];
            const chunk = await toLkChunk(message);
            if (chunk) this.queue.put(chunk);
          } else if (event === "custom") {
            const event = data as VoiceEvent;
            if (event.type === "say") {
              const lkChunk = await toLkChunk(
                new AIMessageChunk({ content: event.data.source })
              );
              if (lkChunk) this.queue.put(lkChunk);
            }
          }
        }
      } catch (err) {
        if (!isGraphInterrupt(err)) throw err;
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
}

export class LangGraphAdapter extends llm.LLM {
  #graph: AnyPregelInterface;
  #options: LangGraphOptions | undefined;

  constructor(graph: AnyPregelInterface, options?: LangGraphOptions) {
    super();
    this.#graph = graph;
    this.#options = options;
  }

  chat(params: {
    chatCtx: llm.ChatContext;
    fncCtx: llm.FunctionContext;
  }): llm.LLMStream {
    const { chatCtx, fncCtx } = params;
    return new LangGraphStream(this, this.#graph, chatCtx, fncCtx, {
      configurable: this.#options?.configurable,
      messageKey: this.#options?.messageKey,
    });
  }
}

function isLkChatImage(i: llm.ChatContent): i is llm.ChatImage {
  return typeof i === "object" && "image" in i && i.image != null;
}

async function toLangChainMessage(m: llm.ChatMessage): Promise<{
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

          throw new Error(
            `Unsupported LiveKit image type: ${JSON.stringify(c)}`
          );
        }

        throw new Error(
          `Unsupported LiveKit content type: ${JSON.stringify(c)}`
        );
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
