# LangGraph x LiveKit Agents

This is the Python implementation of LangGraph LiveKit Agents, which enables building voice-enabled AI agents using LangGraph and LiveKit.

## Initial Setup

1. Install the package as an editable dependency:

```bash
uv pip install -e .
```

2. Configure environment variables:
   Create a `.env` file with your API keys:

```
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_URL=
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
```

3. Run the LangGraph dev server and the LiveKit Agents worker:

```bash
make start-agent
make start-voice
```

## Usage

```python
from langgraph_livekit_agents import LangGraphAdapter
from langgraph.pregel.remote import RemoteGraph

graph = RemoteGraph("agent", url="http://localhost:2024")
agent = pipeline.VoicePipelineAgent(
    vad=ctx.proc.userdata["vad"],
    stt=deepgram.STT(),
    llm=LangGraphAdapter(graph, config={"configurable": {"thread_id": thread_id}}),
    tts=openai.TTS(),
)
```

## Explanation

The LangGraph LiveKit adapter is a wrapper around `llm.LLM`, that maps the LangGraph `messages` stream mode to LiveKit's voice chunks.

The sample also provides utilities for implementing human-in-the-loop-style interrupts and manual `say()` method for playing static messages, built on top of LangGraph's `custom` stream mode. See the `example/pipeline.py` file for more details.

```python
from langgraph.types import interrupt

name, name_msgs = interrupt("What is your name?")

livekit = TypedLivekit(writer)
livekit.say("Give me a second to think...")
```
