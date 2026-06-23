"""Gemini adapter — google-genai `generateContent` streaming.

Maps the normalized request to Gemini Content/Tool/ThinkingConfig, streams
text + thought-summary + function-call parts as normalized Events, and
reconstructs the assistant turn. Like the OpenAI Responses path, the assistant
turn carries a verbatim Gemini Content (as a Raw part) so `thought_signature`s
ride back unchanged on the tool round-trip — Gemini requires this for
multi-step function calling. The SDK is imported lazily so the copilot package
imports without google-genai present. See docs/COPILOT_PROVIDERS.md §4.5 (P4).
"""
from __future__ import annotations

import base64
import json
import os
from typing import AsyncIterator, Optional

from ..types import (
    AssistantMessage, Capabilities, Document, Done, Event, Image, Msg, Raw,
    SystemBlock, Text, Thinking, ThinkingDelta, TextDelta, ToolCall,
    ToolCallStarted, ToolDef, ToolResult, Usage,
)

# our effort scale → Gemini thinking_budget tokens (-1 = dynamic / model decides)
_BUDGET = {"low": 2048, "medium": 8192, "high": -1, "xhigh": -1, "max": -1}

_RAW_GEMINI = "__gemini_content__"   # marks a Raw part holding a verbatim Content


def _b(data_b64: str) -> bytes:
    return base64.b64decode(data_b64)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("METAGEN_GOOGLE_KEY")

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(tools=True, tool_streaming=True, parallel_tools=True,
                            native_pdf=True, native_images=True,
                            reasoning="budget", returns_cot=True)

    # -- request mapping ----------------------------------------------------
    def _user_parts(self, parts, gt) -> list:
        out = []
        for p in parts:
            if isinstance(p, Text) and p.text:
                out.append(gt.Part(text=p.text))
            elif isinstance(p, Image):
                out.append(gt.Part.from_bytes(data=_b(p.data_b64), mime_type=p.media_type))
            elif isinstance(p, Document):
                out.append(gt.Part.from_bytes(data=_b(p.data_b64), mime_type=p.media_type))
            elif isinstance(p, Raw):
                # an Anthropic-shaped base64 image carried through from the UI
                b = p.block
                if b.get("type") == "image" and isinstance(b.get("source"), dict):
                    s = b["source"]
                    if s.get("type") == "base64":
                        out.append(gt.Part.from_bytes(
                            data=_b(s.get("data", "")),
                            mime_type=s.get("media_type", "image/png")))
        return out

    def _to_contents(self, messages: list[Msg], gt) -> list:
        contents = []
        id_to_name: dict[str, str] = {}
        for m in messages:
            if m.role == "assistant":
                for p in m.parts:
                    if isinstance(p, ToolCall):
                        id_to_name[p.id] = p.name
                raw = next((p for p in m.parts
                            if isinstance(p, Raw) and _RAW_GEMINI in p.block), None)
                if raw is not None:                       # verbatim replay (signatures)
                    contents.append(gt.Content.model_validate(raw.block[_RAW_GEMINI]))
                    continue
                parts = []
                for p in m.parts:
                    if isinstance(p, Text) and p.text:
                        parts.append(gt.Part(text=p.text))
                    elif isinstance(p, ToolCall):
                        parts.append(gt.Part(function_call={"id": p.id, "name": p.name,
                                                            "args": p.input}))
                if parts:
                    contents.append(gt.Content(role="model", parts=parts))
                continue
            # user role: tool results become function_response parts
            tool_results = [p for p in m.parts if isinstance(p, ToolResult)]
            others = [p for p in m.parts if not isinstance(p, ToolResult)]
            up = self._user_parts(others, gt)
            if up:
                contents.append(gt.Content(role="user", parts=up))
            for tr in tool_results:
                try:
                    resp = json.loads(tr.content)
                except (json.JSONDecodeError, TypeError):
                    resp = tr.content
                if not isinstance(resp, dict):
                    resp = {"result": resp}
                contents.append(gt.Content(role="user", parts=[gt.Part.from_function_response(
                    name=id_to_name.get(tr.tool_call_id, "tool"), response=resp)]))
        return contents

    # -- stream -------------------------------------------------------------
    async def stream(self, *, model: str, system: list[SystemBlock],
                     messages: list[Msg], tools: list[ToolDef],
                     effort: Optional[str], max_tokens: int) -> AsyncIterator[Event]:
        try:
            from google import genai
            from google.genai import types as gt
        except Exception:  # noqa: BLE001
            yield Done(stop_reason="error")
            return
        if not self._api_key:
            yield Done(stop_reason="error")
            return

        client = genai.Client(api_key=self._api_key)
        sys_text = "\n\n".join(b.text for b in system) or None
        contents = self._to_contents(messages, gt)

        cfg_kwargs: dict = dict(max_output_tokens=max_tokens)
        if sys_text:
            cfg_kwargs["system_instruction"] = sys_text
        if tools:
            cfg_kwargs["tools"] = [gt.Tool(function_declarations=[
                gt.FunctionDeclaration(name=t.name, description=t.description,
                                       parameters=t.schema) for t in tools])]
        want_think = bool(effort) and effort != "off"
        if want_think:
            cfg_kwargs["thinking_config"] = gt.ThinkingConfig(
                include_thoughts=True, thinking_budget=_BUDGET.get(effort, -1))
        config = gt.GenerateContentConfig(**cfg_kwargs)

        text_buf: list[str] = []
        think_buf: list[str] = []
        calls: list[dict] = []           # {id,name,args,signature}
        started: set[str] = set()
        finish_reason: Optional[str] = None
        usage = None

        stream = await client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config)
        async for chunk in stream:
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            for cand in (chunk.candidates or []):
                if getattr(cand, "finish_reason", None):
                    finish_reason = str(cand.finish_reason)
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                for part in (content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        cid = getattr(fc, "id", None) or f"call_{len(calls)}"
                        calls.append({"id": cid, "name": fc.name,
                                      "args": dict(fc.args or {}),
                                      "signature": getattr(part, "thought_signature", None)})
                        if cid not in started:
                            started.add(cid)
                            yield ToolCallStarted(cid, fc.name)
                        continue
                    txt = getattr(part, "text", None)
                    if not txt:
                        continue
                    if getattr(part, "thought", False):
                        think_buf.append(txt)
                        yield ThinkingDelta(txt)
                    else:
                        text_buf.append(txt)
                        yield TextDelta(txt)

        # reconstruct assistant turn
        parts: list = []
        if think_buf:
            sig = next((c["signature"] for c in calls if c["signature"]), None)
            parts.append(Thinking(text="".join(think_buf), signature=sig))
        if text_buf:
            parts.append(Text("".join(text_buf)))
        for c in calls:
            parts.append(ToolCall(id=c["id"], name=c["name"], input=c["args"]))

        # verbatim replay Content (non-thought parts; thought_signature kept on
        # the function_call parts so multi-step tool calling round-trips).
        replay_parts: list = []
        if text_buf:
            replay_parts.append({"text": "".join(text_buf)})
        for c in calls:
            rp: dict = {"function_call": {"id": c["id"], "name": c["name"], "args": c["args"]}}
            if c["signature"]:
                rp["thought_signature"] = c["signature"]
            replay_parts.append(rp)
        if replay_parts:
            parts.append(Raw(block={_RAW_GEMINI: {"role": "model", "parts": replay_parts}}))

        stop = ("tool_use" if calls else
                "length" if finish_reason and "MAX_TOKENS" in finish_reason else
                "end_turn")
        yield AssistantMessage(parts=parts, stop_reason=stop, usage=_usage(usage))


def _usage(u) -> Usage:
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "prompt_token_count", None),
        output_tokens=getattr(u, "candidates_token_count", None),
        thinking_tokens=getattr(u, "thoughts_token_count", None))
