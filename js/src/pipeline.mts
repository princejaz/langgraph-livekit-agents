import type { JobProcess } from "@livekit/agents";
import {
  AutoSubscribe,
  type JobContext,
  WorkerOptions,
  cli,
} from "@livekit/agents";
import * as openai from "@livekit/agents-plugin-openai";
import * as silero from "@livekit/agents-plugin-silero";
import { fileURLToPath } from "node:url";
import { LangGraphAgent } from "./runtime.mjs";
import { randomUUID } from "node:crypto";

import { graph } from "../examples/graph.mjs";

export default {
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    const vad = ctx.proc.userData.vad! as silero.VAD;

    await ctx.connect(undefined, AutoSubscribe.AUDIO_ONLY);
    const participant = await ctx.waitForParticipant();

    const sessionId = (await ctx.room.getSid()) ?? randomUUID();
    console.log(
      `Starting assistant example agent for ${participant.identity} with the session id ${sessionId}`
    );

    const agent = new LangGraphAgent(
      vad,
      openai.STT.withGroq({
        model: "whisper-large-v3-turbo",
        detectLanguage: true,
      }),
      graph,
      new openai.TTS(),
      { configurable: { thread_id: sessionId } }
    );

    agent.start(ctx.room, participant);
  },
};

cli.runApp(new WorkerOptions({ agent: fileURLToPath(import.meta.url) }));
