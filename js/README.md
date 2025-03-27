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

// Can either be a local compiled graph or a deployed graph
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

The sample provides several key features:

1. **Interrupts**: Implement human-in-the-loop-style interrupts using LangGraph's `custom` stream mode. This allows the agent to pause and wait for user input at specific points in the conversation.

2. **Manual Voice Speakout**: Use the `say()` method to play static messages or announcements at any point in the conversation.

Here's an example of using these features:

```typescript
import { typedLiveKit } from "./src/types.mjs";

const liveKit = typedLiveKit(config);

// Using interrupts to get user input
const { content, messages } = liveKit.interrupt("What is your name?");

// Manual voice speakout
liveKit.say("Give me a second to think...");
```

`LangGraphAdapter` supports both graphs deployed in LangGraph Platform and standalone graphs running within LiveKit Agents worker, just swap the `RemoteGraph` with your compiled graph.

```typescript
import { LangGraphAdapter } from "./src/runtime.mts";
import { StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(...).compile();

const agent = new pipeline.VoicePipelineAgent(
  vad,
  openai.STT.withGroq({ model: "whisper-large-v3-turbo" }),
  new LangGraphAdapter(graph),
  new openai.TTS()
);
```
