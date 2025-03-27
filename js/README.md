# LangGraph.js x LiveKit Agents

This is the JavaScript implementation of LangGraph LiveKit Agents, which enables building voice-enabled AI agents using LangGraph and LiveKit.

## Initial Setup

1. Install dependencies:

```bash
pnpm install
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

3. Run the LangGraph.js dev server and the LiveKit Agents worker:

```bashs
pnpm run agent
pnpm run voice
```

## Usage

```typescript
import { LangGraphAdapter } from "./src/runtime.mts";
import { RemoteGraph } from "@langchain/langgraph/remote";

const graph = new RemoteGraph({
  graphId: "agent",
  url: "http://localhost:2024",
});

const agent = new pipeline.VoicePipelineAgent(
  vad,
  openai.STT.withGroq({ model: "whisper-large-v3-turbo" }),
  new LangGraphAdapter(graph),
  new openai.TTS()
);
```

## Explanation

The LangGraph LiveKit adapter is a wrapper around `llm.LLM`, that maps the LangGraph `messages` stream mode to LiveKit's voice chunks.

The sample also provides utilities for implementing human-in-the-loop-style interrupts and manual `say()` method for playing static messages, built on top of LangGraph's `custom` stream mode.

```typescript
const liveKit = typedLiveKit(config);
const name = liveKit.interrupt("What is your name?");

liveKit.say("Give me a second to think...");
```
