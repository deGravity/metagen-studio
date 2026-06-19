"""Provider-neutral agent loop.

Drives a Provider over a tool registry: streams the model, relays normalized
Events to the caller, executes tool calls via their handlers (with the host's
ToolEnv), feeds results back, and loops to max_turns. Knows nothing about
Anthropic, FastAPI, or the studio — the only host coupling is the injected
ToolEnv and tool handlers. See docs/COPILOT_PROVIDERS.md §4.6.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

from .providers.base import Provider
from .tools import ToolEnv, ToolOutcome, ToolRegistry
from .types import (
    AssistantMessage, Done, ErrorEvent, Event, Msg, SystemBlock, ToolCall,
    ToolResult, ToolResultEvent,
)


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
    ) -> AsyncIterator[Event]:
        """Yield the normalized event stream for one user turn (which may span
        several model calls if tools are used). `messages` is the full transcript
        prefix; it is copied, not mutated."""
        history = list(messages)

        for _turn in range(self.max_turns):
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

            history.append(Msg("assistant", assistant.parts))
            tool_calls = [p for p in assistant.parts if isinstance(p, ToolCall)]
            if assistant.stop_reason != "tool_use" or not tool_calls:
                yield Done(stop_reason=assistant.stop_reason)
                return

            # execute tools, feed results back
            results: list[ToolResult] = []
            for tc in tool_calls:
                tool = self.tools.get(tc.name)
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
                yield ToolResultEvent(tool_call_id=tc.id, name=tc.name, result=outcome.result)
                for art in outcome.artifacts:
                    yield art
                results.append(ToolResult(tool_call_id=tc.id,
                                          content=json.dumps(outcome.result),
                                          is_error=outcome.error))
            history.append(Msg("user", results))

        yield ErrorEvent(message=f"tool-use loop exceeded {self.max_turns} turns")
