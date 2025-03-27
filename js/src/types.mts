import { interrupt, type LangGraphRunnableConfig } from "@langchain/langgraph";

export const typedLiveKit = (config: LangGraphRunnableConfig) => ({
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
