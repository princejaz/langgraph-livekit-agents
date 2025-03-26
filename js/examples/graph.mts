import { ChatOpenAI } from "@langchain/openai";
import { LangGraphRunnableConfig, StateGraph } from "@langchain/langgraph";

import {
  MessagesAnnotation,
  interrupt,
  MemorySaver,
} from "@langchain/langgraph";
const checkpointer = new MemorySaver();

const withVoice = (config: LangGraphRunnableConfig) => {
  return {
    say: (
      source: string,
      allowInterruptions: boolean = true,
      addToChatCtx: boolean = false
    ) => {
      config.writer?.({
        type: "say-async",
        data: {
          source,
          allowInterruptions,
          addToChatCtx,
        },
      });
    },
  };
};

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state, config) => {
    const title = interrupt("Tell me first what do you want to do");
    const description = interrupt("Now tell me more about how many things");

    withVoice(config).say("Give me a second to think...");

    return {
      messages: await new ChatOpenAI({
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
            `${title} and description: ${description}`,
        },
        ...state.messages,
      ]),
    };
  })
  .addEdge("__start__", "agent")
  .compile({ checkpointer });
