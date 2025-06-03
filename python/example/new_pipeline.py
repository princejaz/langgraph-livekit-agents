from dotenv import load_dotenv
import logging
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, AutoSubscribe
from livekit.plugins import (
    openai,
    silero,
)
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
from langgraph_livekit_agents import LangGraphAdapter
from langgraph.pregel.remote import RemoteGraph
from uuid import uuid4, uuid5, UUID

load_dotenv()
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

def get_thread_id(sid: str | None) -> str:
    NAMESPACE = UUID("41010b5d-5447-4df5-baf2-97d69f2e9d06")
    if sid is not None:
        return str(uuid5(NAMESPACE, sid))
    return str(uuid4())


async def entrypoint(ctx: agents.JobContext):
    graph = RemoteGraph("agent", url="http://localhost:2024")

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    thread_id = get_thread_id(participant.sid)

    logger.info(
        f"starting voice assistant for participant {participant.identity} (thread ID: {thread_id})"
    )

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        # llm=LangGraphAdapter(graph, config={"configurable": {"thread_id": thread_id}}),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            # noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))