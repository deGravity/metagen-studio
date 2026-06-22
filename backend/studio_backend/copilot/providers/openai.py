"""OpenAI-style adapter — one class, two wire modes.

* ``mode="responses"``  → OpenAI Responses API (GPT-5.x): best tool +
  reasoning story; native PDF/images; ``reasoning.effort`` + summarized CoT.
* ``mode="chat_completions"`` → OpenAI **Chat Completions** surface, which is
  what our **vLLM** server speaks (Qwen3.x, gpt-oss-120b). Same SDK pointed at
  a ``base_url``; reasoning arrives as ``reasoning_content`` deltas (vLLM
  ``--reasoning-parser``) and tool calls as the standard streamed
  ``tool_calls`` format.

Both paths translate the normalized request (SystemBlocks, Msg history,
ToolDefs, effort) into the vendor shape, stream it, and yield normalized
Events — TextDelta / ThinkingDelta / ToolCallStarted, then exactly one
AssistantMessage (reconstructed parts + stop_reason + usage). The engine runs
the tool loop; this adapter knows nothing about it. See
docs/COPILOT_PROVIDERS.md §4.2 / §4.5 / §7 (P2).
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from ..types import (
    AssistantMessage, Capabilities, Document, Done, Event, Image, Msg, Raw,
    SystemBlock, Text, Thinking, ThinkingDelta, TextDelta, ToolCall,
    ToolCallStarted, ToolDef, ToolResult, Usage,
)

# Our effort scale is {off,low,medium,high,xhigh,max}; OpenAI's reasoning.effort
# is {minimal,low,medium,high}. Collapse the top of our scale onto 'high'.
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high",
               "xhigh": "high", "max": "high"}


def _map_effort(effort: Optional[str]) -> Optional[str]:
    if not effort or effort == "off":
        return None
    return _EFFORT_MAP.get(effort, "high")


def _data_url(media_type: str, data_b64: str) -> str:
    return f"data:{media_type};base64,{data_b64}"


class OpenAIProvider:
    """OpenAI / OpenAI-compatible adapter. ``profile`` carries per-model
    capability + parser hints (used mainly for vLLM open models):
    ``{reasoning: 'effort'|'toggle'|'none', native_pdf: bool,
       native_images: bool}``."""

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 mode: str = "responses", profile: Optional[dict] = None):
        self.mode = mode
        self.profile = profile or {}
        self.name = "openai" if mode == "responses" else "openai-compat"
        # vLLM commonly wants *some* key string even when auth is off.
        self._client = AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    # -- capabilities -------------------------------------------------------
    def capabilities(self, model: str) -> Capabilities:
        if self.mode == "responses":
            return Capabilities(
                tools=True, tool_streaming=True, parallel_tools=True,
                native_pdf=True, native_images=True, reasoning="effort",
                returns_cot=True)
        p = self.profile
        reasoning = p.get("reasoning", "toggle")
        return Capabilities(
            tools=True, tool_streaming=True, parallel_tools=True,
            native_pdf=bool(p.get("native_pdf", False)),
            native_images=bool(p.get("native_images", False)),
            reasoning=reasoning if reasoning in ("effort", "toggle", "none") else "none",
            returns_cot=reasoning != "none")

    # -- entry point --------------------------------------------------------
    def stream(self, *, model: str, system: list[SystemBlock], messages: list[Msg],
               tools: list[ToolDef], effort: Optional[str],
               max_tokens: int) -> AsyncIterator[Event]:
        if self.mode == "responses":
            return self._stream_responses(model, system, messages, tools, effort, max_tokens)
        return self._stream_chat(model, system, messages, tools, effort, max_tokens)

    # ======================================================================
    # Chat Completions mode (vLLM: Qwen3.x, gpt-oss-120b)
    # ======================================================================
    def _chat_content(self, parts) -> object:
        """Text/Image parts -> Chat Completions content (str if plain text,
        else the multimodal list). Returns None if there is nothing to send."""
        has_media = any(isinstance(p, (Image, Document, Raw)) for p in parts)
        if not has_media:
            txt = "".join(p.text for p in parts if isinstance(p, Text))
            return txt if txt else None
        out: list[dict] = []
        for p in parts:
            if isinstance(p, Text):
                if p.text:
                    out.append({"type": "text", "text": p.text})
            elif isinstance(p, Image):
                out.append({"type": "image_url",
                            "image_url": {"url": _data_url(p.media_type, p.data_b64)}})
            elif isinstance(p, Raw):
                # Best-effort carry-through of an Anthropic-shaped image block
                # (e.g. from a prior turn). PDFs are handled by the P3 pipeline.
                b = p.block
                if b.get("type") == "image" and isinstance(b.get("source"), dict):
                    src = b["source"]
                    if src.get("type") == "base64":
                        out.append({"type": "image_url", "image_url": {
                            "url": _data_url(src.get("media_type", "image/png"),
                                             src.get("data", ""))}})
            # Document parts intentionally skipped here (PDF pipeline = P3).
        return out or None

    def _to_chat_messages(self, system, messages) -> list[dict]:
        out: list[dict] = []
        sys_text = "\n\n".join(b.text for b in system)
        if sys_text:
            out.append({"role": "system", "content": sys_text})
        for m in messages:
            if m.role == "assistant":
                content = "".join(p.text for p in m.parts if isinstance(p, Text))
                tcs = [{"id": p.id, "type": "function",
                        "function": {"name": p.name,
                                     "arguments": json.dumps(p.input)}}
                       for p in m.parts if isinstance(p, ToolCall)]
                msg: dict = {"role": "assistant", "content": content or None}
                if tcs:
                    msg["tool_calls"] = tcs
                out.append(msg)
                continue
            # user role: tool results become their own `tool` messages; the
            # rest (text/images) become a single user message.
            tool_results = [p for p in m.parts if isinstance(p, ToolResult)]
            others = [p for p in m.parts if not isinstance(p, ToolResult)]
            content = self._chat_content(others)
            if content is not None:
                out.append({"role": "user", "content": content})
            for tr in tool_results:
                out.append({"role": "tool", "tool_call_id": tr.tool_call_id,
                            "content": tr.content})
        return out

    async def _stream_chat(self, model, system, messages, tools, effort, max_tokens):
        api_messages = self._to_chat_messages(system, messages)
        api_tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.schema}}
            for t in tools]

        kwargs: dict = dict(model=model, messages=api_messages,
                            max_tokens=max_tokens, stream=True,
                            stream_options={"include_usage": True})
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"

        # Reasoning: Qwen3 toggles via chat_template_kwargs.enable_thinking;
        # gpt-oss takes reasoning_effort. Both ride in extra_body so we don't
        # depend on SDK-version field support.
        reasoning = self.profile.get("reasoning", "toggle")
        eff = _map_effort(effort)
        extra_body: dict = {}
        if reasoning == "toggle":
            extra_body["chat_template_kwargs"] = {"enable_thinking": eff is not None}
        elif reasoning == "effort" and eff is not None:
            extra_body["reasoning_effort"] = eff
        if extra_body:
            kwargs["extra_body"] = extra_body

        text_buf: list[str] = []
        think_buf: list[str] = []
        tool_accum: dict[int, dict] = {}
        started: set[int] = set()
        finish_reason: Optional[str] = None
        usage_obj = None

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            delta = ch.delta
            if ch.finish_reason:
                finish_reason = ch.finish_reason
            if delta is None:
                continue
            # vLLM exposes streamed reasoning under either `reasoning_content`
            # (older builds) or `reasoning` (newer Qwen3 builds) — accept both.
            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                think_buf.append(rc)
                yield ThinkingDelta(rc)
            if delta.content:
                text_buf.append(delta.content)
                yield TextDelta(delta.content)
            for tc in (delta.tool_calls or []):
                idx = tc.index
                e = tool_accum.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.id:
                    e["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        e["name"] = fn.name
                    if fn.arguments:
                        e["args"] += fn.arguments
                if idx not in started and e["name"]:
                    started.add(idx)
                    yield ToolCallStarted(e["id"] or f"call_{idx}", e["name"])

        parts: list = []
        if think_buf:
            parts.append(Thinking(text="".join(think_buf)))
        if text_buf:
            parts.append(Text("".join(text_buf)))
        for idx in sorted(tool_accum):
            e = tool_accum[idx]
            if not e["name"]:
                continue
            try:
                args = json.loads(e["args"]) if e["args"].strip() else {}
            except json.JSONDecodeError:
                args = {"__raw_arguments__": e["args"]}
            parts.append(ToolCall(id=e["id"] or f"call_{idx}", name=e["name"], input=args))

        stop_reason = "tool_use" if (finish_reason == "tool_calls" or
                                     any(isinstance(p, ToolCall) for p in parts)) \
            else ("end_turn" if finish_reason in ("stop", None) else finish_reason)
        yield AssistantMessage(parts=parts, stop_reason=stop_reason,
                               usage=_chat_usage(usage_obj))

    # ======================================================================
    # Responses mode (OpenAI GPT-5.x)
    # ======================================================================
    def _to_responses_input(self, messages) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "assistant":
                content = [{"type": "output_text", "text": p.text}
                           for p in m.parts if isinstance(p, Text) and p.text]
                if content:
                    out.append({"role": "assistant", "content": content})
                for p in m.parts:
                    if isinstance(p, Raw):
                        # Replayed reasoning / native items captured verbatim
                        # during reconstruction (kept in output order).
                        out.append(dict(p.block))
                    elif isinstance(p, ToolCall):
                        out.append({"type": "function_call", "call_id": p.id,
                                    "name": p.name, "arguments": json.dumps(p.input)})
                continue
            # user role
            content: list[dict] = []
            for p in m.parts:
                if isinstance(p, Text) and p.text:
                    content.append({"type": "input_text", "text": p.text})
                elif isinstance(p, Image):
                    content.append({"type": "input_image",
                                    "image_url": _data_url(p.media_type, p.data_b64)})
                elif isinstance(p, Document):
                    content.append({"type": "input_file", "filename": p.name or "doc.pdf",
                                    "file_data": _data_url(p.media_type, p.data_b64)})
                elif isinstance(p, Raw):
                    out.append(dict(p.block))
            if content:
                out.append({"role": "user", "content": content})
            for tr in (p for p in m.parts if isinstance(p, ToolResult)):
                out.append({"type": "function_call_output", "call_id": tr.tool_call_id,
                            "output": tr.content})
        return out

    async def _stream_responses(self, model, system, messages, tools, effort, max_tokens):
        instructions = "\n\n".join(b.text for b in system)
        api_input = self._to_responses_input(messages)
        api_tools = [{"type": "function", "name": t.name,
                      "description": t.description, "parameters": t.schema}
                     for t in tools]

        kwargs: dict = dict(model=model, input=api_input, instructions=instructions,
                            max_output_tokens=max_tokens, store=False)
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"
        eff = _map_effort(effort)
        if eff is not None:
            kwargs["reasoning"] = {"effort": eff, "summary": "auto"}
            # encrypted reasoning so it can be replayed statelessly on the
            # tool round-trip (store=False -> no server-side state).
            kwargs["include"] = ["reasoning.encrypted_content"]

        async with self._client.responses.stream(**kwargs) as stream:
            async for event in stream:
                et = getattr(event, "type", "")
                if et == "response.output_text.delta":
                    yield TextDelta(event.delta)
                elif et in ("response.reasoning_summary_text.delta",
                            "response.reasoning_text.delta"):
                    yield ThinkingDelta(event.delta)
                elif et == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", "") == "function_call":
                        yield ToolCallStarted(getattr(item, "call_id", "") or
                                              getattr(item, "id", ""),
                                              getattr(item, "name", ""))
            final = await stream.get_final_response()

        parts: list = []
        has_tool = False
        for item in (final.output or []):
            it = getattr(item, "type", "")
            if it == "message":
                txt = "".join(getattr(c, "text", "") for c in getattr(item, "content", [])
                              if getattr(c, "type", "") == "output_text")
                if txt:
                    parts.append(Text(txt))
            elif it == "reasoning":
                # keep the full item (incl. id + encrypted_content) for replay
                parts.append(Raw(block=_to_plain(item)))
            elif it == "function_call":
                has_tool = True
                try:
                    args = json.loads(item.arguments) if item.arguments else {}
                except json.JSONDecodeError:
                    args = {"__raw_arguments__": item.arguments}
                parts.append(ToolCall(id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                                      name=item.name, input=args))

        status = getattr(final, "status", None)
        stop_reason = "tool_use" if has_tool else (
            "length" if status == "incomplete" else "end_turn")
        yield AssistantMessage(parts=parts, stop_reason=stop_reason,
                               usage=_responses_usage(getattr(final, "usage", None)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_plain(obj) -> dict:
    """SDK model -> plain dict (for verbatim replay)."""
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(exclude_none=True) if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    return dict(obj)


def _chat_usage(u) -> Usage:
    if u is None:
        return Usage()
    det = getattr(u, "completion_tokens_details", None)
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", None),
        output_tokens=getattr(u, "completion_tokens", None),
        thinking_tokens=getattr(det, "reasoning_tokens", None) if det else None)


def _responses_usage(u) -> Usage:
    if u is None:
        return Usage()
    det = getattr(u, "output_tokens_details", None)
    return Usage(
        input_tokens=getattr(u, "input_tokens", None),
        output_tokens=getattr(u, "output_tokens", None),
        thinking_tokens=getattr(det, "reasoning_tokens", None) if det else None)
