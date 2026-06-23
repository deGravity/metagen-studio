"""Anthropic adapter — maps the normalized request to the Messages API and
streams normalized Events. Encapsulates everything Anthropic-specific that used
to live inline in chat.py (content blocks, adaptive thinking, tool format)."""
from __future__ import annotations

import os
from typing import AsyncIterator, Optional

from ..types import (
    AssistantMessage, Capabilities, Done, Event, Image, Document, Msg, Raw,
    SystemBlock, Text, Thinking, ThinkingDelta, TextDelta, ToolCall, ToolCallStarted,
    ToolDef, ToolResult, Usage,
)

_PDF_BETA = "files-api-2025-04-14"


def _part_to_block(p) -> dict:
    if isinstance(p, Text):
        return {"type": "text", "text": p.text}
    if isinstance(p, Image):
        return {"type": "image", "source": {"type": "base64",
                "media_type": p.media_type, "data": p.data_b64}}
    if isinstance(p, Document):
        return {"type": "document", "source": {"type": "base64",
                "media_type": p.media_type, "data": p.data_b64}}
    if isinstance(p, ToolCall):
        return {"type": "tool_use", "id": p.id, "name": p.name, "input": p.input}
    if isinstance(p, ToolResult):
        b = {"type": "tool_result", "tool_use_id": p.tool_call_id, "content": p.content}
        if p.is_error:
            b["is_error"] = True
        return b
    if isinstance(p, Thinking):
        if p.redacted is not None:
            return {"type": "redacted_thinking", "data": p.redacted}
        return {"type": "thinking", "thinking": p.text, "signature": p.signature}
    if isinstance(p, Raw):
        return dict(p.block)
    raise TypeError(f"unmappable part: {type(p).__name__}")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("METAGEN_ANTHROPIC_API_KEY")

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(tools=True, tool_streaming=True, parallel_tools=True,
                            native_pdf=True, native_images=True,
                            reasoning="effort", returns_cot=True)

    async def stream(self, *, model: str, system: list[SystemBlock],
                     messages: list[Msg], tools: list[ToolDef],
                     effort: Optional[str], max_tokens: int) -> AsyncIterator[Event]:
        if not self._api_key:
            yield Done(stop_reason="error")
            return
        from anthropic import AsyncAnthropic   # lazy: SDK is an optional extra
        client = AsyncAnthropic(api_key=self._api_key)

        sys_blocks = []
        for b in system:
            blk = {"type": "text", "text": b.text}
            if b.cache:
                blk["cache_control"] = {"type": "ephemeral"}
            sys_blocks.append(blk)

        api_messages = [{"role": m.role, "content": [_part_to_block(p) for p in m.parts]}
                        for m in messages]
        api_tools = [{"name": t.name, "description": t.description,
                      "input_schema": t.schema} for t in tools]

        want_think = bool(effort) and effort != "off"
        if want_think:
            thinking = {"type": "adaptive", "display": "summarized"}
            output_config = {"effort": effort}
        else:
            thinking = {"type": "disabled"}
            output_config = None

        kwargs = dict(model=model, max_tokens=max_tokens, messages=api_messages,
                      system=sys_blocks, tools=api_tools, thinking=thinking,
                      extra_headers={"anthropic-beta": _PDF_BETA})
        if output_config:
            kwargs["output_config"] = output_config

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                et = getattr(event, "type", None)
                if et == "text":
                    yield TextDelta(event.text)
                elif et == "thinking":
                    yield ThinkingDelta(getattr(event, "thinking", ""))
                elif et == "content_block_start":
                    blk = event.content_block
                    if blk.type == "tool_use":
                        yield ToolCallStarted(blk.id, blk.name)
            final = await stream.get_final_message()

        parts: list = []
        for b in final.content:
            if b.type == "text":
                parts.append(Text(b.text))
            elif b.type == "thinking":
                parts.append(Thinking(text=b.thinking, signature=getattr(b, "signature", None)))
            elif b.type == "redacted_thinking":
                parts.append(Thinking(redacted=getattr(b, "data", "")))
            elif b.type == "tool_use":
                parts.append(ToolCall(id=b.id, name=b.name, input=b.input))
        u = getattr(final, "usage", None)
        det = getattr(u, "output_tokens_details", None) if u else None
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", None) if u else None,
            output_tokens=getattr(u, "output_tokens", None) if u else None,
            thinking_tokens=getattr(det, "thinking_tokens", None) if det else None,
        )
        yield AssistantMessage(parts=parts, stop_reason=final.stop_reason, usage=usage)
