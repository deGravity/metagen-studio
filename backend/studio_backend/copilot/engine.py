"""Provider-neutral agent loop.

Drives a Provider over a tool registry: streams the model, relays normalized
Events to the caller, executes tool calls via their handlers (with the host's
ToolEnv), feeds results back, and loops to max_turns. Knows nothing about
Anthropic, FastAPI, or the studio — the only host coupling is the injected
ToolEnv and tool handlers. See docs/COPILOT_PROVIDERS.md §4.6.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Callable, Optional

from .providers.base import Provider
from .tools import ToolEnv, ToolOutcome, ToolRegistry
from .types import (
    AssistantMessage, Done, ErrorEvent, Event, Msg, SystemBlock, ToolCall,
    ToolResult, ToolResultEvent, msg_to_dict, part_to_dict,
)

# log hook: log(event_type, json-able payload). The host turns these into
# transcript/session events; the engine stays storage-agnostic.
LogFn = Callable[[str, dict], None]


class CopilotEngine:
    def __init__(self, provider: Provider, tools: ToolRegistry, max_turns: int = 8):
        self.provider = provider
        self.tools = tools
        self.max_turns = max_turns

    async def run(
        self,
        *,
        model: str,
        system: list[SystemBlock],
        messages: list[Msg],
        env: ToolEnv,
        effort: Optional[str],
        max_tokens: int,
        log: Optional[LogFn] = None,
    ) -> AsyncIterator[Event]:
        """Yield the normalized event stream for one user turn (which may span
        several model calls if tools are used). `messages` is the full transcript
        prefix; it is copied, not mutated. `log` (optional) receives provider-
        neutral copilot_request/copilot_response payloads per model call."""
        history = list(messages)

        for call_index in range(self.max_turns):
            if log:
                log("copilot_request", {
                    "call_index": call_index, "model": model, "effort": effort,
                    "max_tokens": max_tokens,
                    "system": [b.text for b in system],
                    "messages": [msg_to_dict(m) for m in history],
                    "tools": [t.name for t in self.tools.defs()]})

            assistant: Optional[AssistantMessage] = None
            try:
                async for ev in self.provider.stream(
                    model=model, system=system, messages=history,
                    tools=self.tools.defs(), effort=effort, max_tokens=max_tokens,
                ):
                    if isinstance(ev, AssistantMessage):
                        assistant = ev
                    yield ev   # relay deltas / message / usage to the consumer
            except Exception as exc:  # noqa: BLE001
                import traceback
                yield ErrorEvent(message=f"{type(exc).__name__}: {exc}",
                                 detail=traceback.format_exc())
                return

            if assistant is None:
                yield ErrorEvent(message="provider returned no assistant message")
                return

            if log:
                log("copilot_response", {
                    "call_index": call_index, "stop_reason": assistant.stop_reason,
                    "usage": {"input_tokens": assistant.usage.input_tokens,
                              "output_tokens": assistant.usage.output_tokens,
                              "thinking_tokens": assistant.usage.thinking_tokens},
                    "content_blocks": [part_to_dict(p) for p in assistant.parts]})

            history.append(Msg("assistant", assistant.parts))
            tool_calls = [p for p in assistant.parts if isinstance(p, ToolCall)]
            if assistant.stop_reason != "tool_use" or not tool_calls:
                yield Done(stop_reason=assistant.stop_reason)
                return

            # execute tools, feed results back (artifacts before the result event,
            # so consumers can emit UI ahead of the tool_result)
            results: list[ToolResult] = []
            for tc in tool_calls:
                tool = self.tools.get(tc.name)
                t0 = time.perf_counter()
                if tool is None:
                    outcome = ToolOutcome(result={"error": f"unknown tool: {tc.name}"},
                                          error=True)
                else:
                    try:
                        outcome = await tool.handler(tc.input, env)
                    except Exception as exc:  # noqa: BLE001
                        import traceback
                        outcome = ToolOutcome(
                            result={"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}, error=True)
                elapsed = time.perf_counter() - t0
                for art in outcome.artifacts:
                    art.tool_call_id = tc.id
                    art.tool_name = tc.name
                    yield art
                yield ToolResultEvent(tool_call_id=tc.id, name=tc.name,
                                      result=outcome.result, elapsed_s=round(elapsed, 3))
                results.append(ToolResult(tool_call_id=tc.id,
                                          content=json.dumps(outcome.result),
                                          is_error=outcome.error))
            history.append(Msg("user", results))

        yield ErrorEvent(message=f"tool-use loop exceeded {self.max_turns} turns")
