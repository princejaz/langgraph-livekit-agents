import logging
from uuid import uuid4, uuid5, UUID
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
from langgraph_livekit_agents import LangGraphAdapter
from langgraph.pregel.remote import RemoteGraph

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


def get_thread_id(sid: str | None) -> str:
    NAMESPACE = UUID("41010b5d-5447-4df5-baf2-97d69f2e9d06")
    if sid is not None:
        return str(uuid5(NAMESPACE, sid))
    return str(uuid4())


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    thread_id = get_thread_id(participant.sid)

    logger.info(
        f"starting voice assistant for participant {participant.identity} (thread ID: {thread_id})"
    )

    graph = RemoteGraph("agent", url="http://localhost:2024")
    agent = pipeline.VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=LangGraphAdapter(graph, config={"configurable": {"thread_id": thread_id}}),
        tts=openai.TTS(),
    )

    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
