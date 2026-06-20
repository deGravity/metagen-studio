"""Provider-neutral data model for the copilot engine.

Nothing here imports a vendor SDK, FastAPI, or studio internals — this is the
shared vocabulary the engine, provider adapters, and consumers all speak.
See docs/COPILOT_PROVIDERS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

# --------------------------------------------------------------------------- #
# message content parts
# --------------------------------------------------------------------------- #
@dataclass
class Text:
    text: str


@dataclass
class Image:
    data_b64: str
    media_type: str = "image/png"


@dataclass
class Document:
    """A document attachment (e.g. PDF). Adapters either pass it through
    natively (capability native_pdf) or it is pre-rendered to Text/Image by the
    attachment pipeline before reaching the provider."""
    data_b64: str
    media_type: str = "application/pdf"
    name: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str          # JSON-encoded tool result for the model
    is_error: bool = False


@dataclass
class Thinking:
    """A reasoning block. `text` may be empty (display=omitted) but `signature`
    must be preserved verbatim for providers that require it on tool round-trips."""
    text: str = ""
    signature: Optional[str] = None
    redacted: Optional[str] = None   # opaque redacted_thinking payload


@dataclass
class Raw:
    """A provider-native content block passed through verbatim. Lets P1 carry
    existing frontend attachment blocks (images, Files-API file refs) without
    yet fully normalizing them; the proper attachment model lands in P3."""
    block: dict


Part = Union[Text, Image, Document, ToolCall, ToolResult, Thinking, Raw]


@dataclass
class Msg:
    role: Literal["user", "assistant"]
    parts: list[Part]


@dataclass
class SystemBlock:
    text: str
    cache: bool = False   # adapters that support prompt caching may honor this


@dataclass
class ToolDef:
    name: str
    description: str
    schema: dict          # JSON Schema for the tool input


@dataclass
class Capabilities:
    tools: bool = True
    tool_streaming: bool = True
    parallel_tools: bool = True
    native_pdf: bool = False
    native_images: bool = False
    reasoning: Literal["none", "effort", "budget", "toggle"] = "none"
    returns_cot: bool = False   # does the provider return a (summarized) CoT?


# --------------------------------------------------------------------------- #
# event stream (engine output; provider adapters emit a subset)
# --------------------------------------------------------------------------- #
@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallStarted:
    id: str
    name: str


@dataclass
class AssistantMessage:
    """The fully-reconstructed assistant turn (incl. thinking/tool_use parts),
    plus stop_reason + usage. The engine appends this to the transcript and
    decides whether to continue the tool loop."""
    parts: list[Part]
    stop_reason: str
    usage: "Usage"


@dataclass
class ToolResultEvent:
    tool_call_id: str
    name: str
    result: Any
    elapsed_s: float = 0.0


@dataclass
class Artifact:
    """Sidecar payload a tool wants surfaced (geometry/sim/proposal/...). UI
    consumers render it; headless consumers may ignore it. The engine stamps
    the originating tool's id/name so consumers can correlate without ordering
    assumptions."""
    kind: str
    data: dict
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None


@dataclass
class Done:
    stop_reason: str


@dataclass
class ErrorEvent:
    message: str
    detail: Optional[str] = None


Event = Union[
    TextDelta, ThinkingDelta, ToolCallStarted, AssistantMessage,
    ToolResultEvent, Artifact, Usage, Done, ErrorEvent,
]


# --------------------------------------------------------------------------- #
# JSON serialization (for logging / transcripts) — provider-neutral
# --------------------------------------------------------------------------- #
def part_to_dict(p: Part) -> dict:
    if isinstance(p, Text):
        return {"type": "text", "text": p.text}
    if isinstance(p, Thinking):
        if p.redacted is not None:
            return {"type": "redacted_thinking", "data": p.redacted}
        return {"type": "thinking", "thinking": p.text, "signature": p.signature}
    if isinstance(p, ToolCall):
        return {"type": "tool_use", "id": p.id, "name": p.name, "input": p.input}
    if isinstance(p, ToolResult):
        return {"type": "tool_result", "tool_use_id": p.tool_call_id,
                "content": p.content, "is_error": p.is_error}
    if isinstance(p, Image):
        return {"type": "image", "media_type": p.media_type, "bytes": len(p.data_b64)}
    if isinstance(p, Document):
        return {"type": "document", "media_type": p.media_type, "name": p.name,
                "bytes": len(p.data_b64)}
    if isinstance(p, Raw):
        return {"type": "raw", "block_type": p.block.get("type")}
    return {"type": "unknown"}


def msg_to_dict(m: Msg) -> dict:
    return {"role": m.role, "parts": [part_to_dict(p) for p in m.parts]}
