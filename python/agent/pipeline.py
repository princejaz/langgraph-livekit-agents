import logging
from uuid import uuid4
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    pipeline,
)
from livekit.plugins import openai, deepgram, silero
from runtime import LangGraph
from agent import graph

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    thread_id = participant.sid or str(uuid4())

    agent = pipeline.VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=LangGraph(graph, config={"configurable": {"thread_id": thread_id}}),
        tts=openai.TTS(),
    )

    agent.once("agent_speech_interrupted", lambda evt: logger.info("interrupted"))
    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
