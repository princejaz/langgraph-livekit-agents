import type { JobProcess } from "@livekit/agents";
import {
  type JobContext,
  AutoSubscribe,
  WorkerOptions,
  cli,
  pipeline,
} from "@livekit/agents";
import * as openai from "@livekit/agents-plugin-openai";
import * as silero from "@livekit/agents-plugin-silero";
import { fileURLToPath } from "node:url";
import { LangGraphAdapter } from "../src/runtime.mjs";
import { RemoteGraph } from "@langchain/langgraph/remote";
import { v5 as uuid5, v4 as uuid4 } from "uuid";

const NAMESPACE = "41010b5d-5447-4df5-baf2-97d69f2e9d06";
const getThreadId = (sid: string | undefined) => {
  if (sid != null) return uuid5(sid, NAMESPACE);
  return uuid4();
};

export default {
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    const vad = ctx.proc.userData.vad! as silero.VAD;

    await ctx.connect(undefined, AutoSubscribe.AUDIO_ONLY);
    const participant = await ctx.waitForParticipant();
    const threadId = getThreadId(await ctx.room.getSid());

    console.debug(
      `Starting voice pipeline for ${participant.identity} (thread ID: ${threadId})`
    );

    const graph = new RemoteGraph({
      graphId: "agent",
      url: "http://localhost:2024",
    });

    const agent = new pipeline.VoicePipelineAgent(
      vad,
      openai.STT.withGroq({ model: "whisper-large-v3-turbo" }),
      new LangGraphAdapter(graph, { configurable: { thread_id: threadId } }),
      new openai.TTS()
    );

    agent.start(ctx.room, participant);
  },
};

cli.runApp(new WorkerOptions({ agent: fileURLToPath(import.meta.url) }));
