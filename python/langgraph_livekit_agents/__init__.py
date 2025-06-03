"""
LangGraphAdapter masquerades as an livekit.LLM and translates the LiveKit chat chunks
into LangGraph messages.
"""

from typing import Any, Optional, Dict

# Refactored LiveKit imports
from livekit.agents.llm.llm import LLM, LLMStream, ChatChunk, ChoiceDelta
from livekit.agents.llm.chat_context import ChatContext, ChatMessage, ImageContent
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool, ToolChoice

from langgraph.pregel import PregelProtocol
from langchain_core.messages import BaseMessageChunk, AIMessage, HumanMessage
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import SynthesizeStream
from livekit.agents.utils import shortuuid
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from httpx import HTTPStatusError

import logging

logger = logging.getLogger(__name__)


# https://github.com/livekit/agents/issues/1370#issuecomment-2588821571
class FlushSentinel(str, SynthesizeStream._FlushSentinel):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class LangGraphStream(LLMStream):  # Use direct import
    def __init__(
        self,
        llm_instance: LLM,  # Changed 'llm' to 'llm_instance' to avoid conflict with module alias if it were still present
        chat_ctx: ChatContext, # Use direct import
        graph: PregelProtocol,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions = None,
    ):
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options
        )
        self._graph = graph

    async def _run(self):
        input_human_message = next(
            (
                self._to_message(m)
                for m in reversed(self.chat_ctx.items) # chat_ctx is ChatContext
                if isinstance(m, ChatMessage) and m.role == "user" # Use direct ChatMessage
            ),
            None,
        )

        messages = [input_human_message] if input_human_message else []
        input = {"messages": messages}

        if interrupt := await self._get_interrupt():
            used_messages = [
                AIMessage(interrupt.value),
                input_human_message,
            ]
            input = Command(resume=(input_human_message.content, used_messages))

        try:
            async for mode, data in self._graph.astream(
                input, config=self._llm._config, stream_mode=["messages", "custom"] # _llm here refers to the instance passed to __init__
            ):
                if mode == "messages":
                    if chunk := await self._to_livekit_chunk(data[0]):
                        self._event_ch.send_nowait(chunk)
                if mode == "custom":
                    if isinstance(data, dict) and (event := data.get("type")):
                        if event == "say" or event == "flush":
                            content = (data.get("data") or {}).get("content")
                            if chunk := await self._to_livekit_chunk(content):
                                self._event_ch.send_nowait(chunk)
                            self._event_ch.send_nowait(
                                self._create_livekit_chunk(FlushSentinel())
                            )
        except GraphInterrupt:
            pass

        if interrupt := await self._get_interrupt():
            if chunk := await self._to_livekit_chunk(interrupt.value):
                self._event_ch.send_nowait(chunk)

    async def _get_interrupt(cls) -> Optional[str]:
        try:
            state = await cls._graph.aget_state(config=cls._llm._config)
            interrupts = [
                interrupt for task in state.tasks for interrupt in task.interrupts
            ]
            assistant = next(
                (
                    interrupt
                    for interrupt in reversed(interrupts)
                    if isinstance(interrupt.value, str)
                ),
                None,
            )
            return assistant
        except HTTPStatusError as e:
            return None

    def _to_message(cls, msg: ChatMessage) -> HumanMessage: # Use direct ChatMessage
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = []
            for c in msg.content:
                if isinstance(c, str):
                    content.append({"type": "text", "text": c})
                elif isinstance(c, ImageContent): # Use direct ChatImage
                    if isinstance(c.image, str):
                        content.append({"type": "image_url", "image_url": c.image})
                    else:
                        logger.warning("Unsupported image type")
                else:
                    logger.warning("Unsupported content type")
        else:
            content = ""
        return HumanMessage(content=content, id=msg.id)

    @staticmethod
    def _create_livekit_chunk(
        content: str,
        *,
        id: str | None = None,
    ) -> ChatChunk | None:
        return ChatChunk(
            id=id or shortuuid(),
            delta=ChoiceDelta(role="assistant", content=content)
        )

    @staticmethod
    async def _to_livekit_chunk(
        msg: BaseMessageChunk | str | None,
    ) -> ChatChunk | None: # Use direct ChatChunk
        if not msg:
            return None
        request_id = None
        content_text = None # Renamed for clarity

        if isinstance(msg, str):
            content_text = msg
        elif hasattr(msg, "content") and isinstance(msg.content, str):
            request_id = getattr(msg, "id", None)
            content_text = msg.content
        elif isinstance(msg, dict):
            request_id = msg.get("id")
            content_text = msg.get("content")
        
        if content_text is None:
             return None

        return LangGraphStream._create_livekit_chunk(content_text, id=request_id)


class LangGraphAdapter(LLM): # Use direct import
    def __init__(self, graph: Any, config: dict[str, Any] | None = None):
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(
        self,
        chat_ctx: ChatContext, # Use direct import
        tools: Optional[list[FunctionTool | RawFunctionTool]] = None,
        tool_choice: Optional[ToolChoice | str] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> LLMStream: # Use direct import
        stream_tools = tools if tools is not None else []
        return LangGraphStream(
            self, # This is llm_instance
            chat_ctx=chat_ctx,
            graph=self._graph,
            tools=stream_tools,
            conn_options=conn_options,
        )
