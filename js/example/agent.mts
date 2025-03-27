import { ChatOpenAI } from "@langchain/openai";
import { LangGraphRunnableConfig, StateGraph } from "@langchain/langgraph";
import { MessagesAnnotation, interrupt } from "@langchain/langgraph";

const typedLiveKit = (config: LangGraphRunnableConfig) => ({
  say: (source: string) => config.writer?.({ type: "say", data: { source } }),
  interrupt: interrupt<
    string,
    {
      content: string;
      messages: [
        { type: "ai"; content: string },
        { type: "human"; content: string }
      ];
    }
  >,
});

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state, config) => {
    const liveKit = typedLiveKit(config);
    const name = liveKit.interrupt("What is your name?");

    liveKit.say("Give me a second to think...");

    const response = await new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0,
    }).invoke([
      {
        type: "system",
        content:
          `You are a voice assistant created by LiveKit. ` +
          `Your interface with users will be voice. You should use short and concise responses, ` +
          `and avoiding usage of unpronounceable punctuation.`,
      },
      {
        type: "user",
        content:
          `Tell me a short haiku about the following topic: ` +
          `User name: ${name.content}`,
      },
      ...state.messages,
    ]);

    return {
      messages: [...name.messages, response],
    };
  })
  .addEdge("__start__", "agent")
  .compile();
