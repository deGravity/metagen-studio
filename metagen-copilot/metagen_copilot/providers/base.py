"""Provider adapter interface.

An adapter maps the normalized request (system blocks, Msg history, ToolDefs,
effort) to a vendor API, streams the response, and yields normalized Events —
TextDelta / ThinkingDelta / ToolCallStarted for the live UI, then exactly one
AssistantMessage (the reconstructed turn + stop_reason + usage). The engine,
not the adapter, runs the tool loop.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from ..types import Capabilities, Event, Msg, SystemBlock, ToolDef


@runtime_checkable
class Provider(Protocol):
    name: str

    def capabilities(self, model: str) -> Capabilities:
        ...

    def stream(
        self,
        *,
        model: str,
        system: list[SystemBlock],
        messages: list[Msg],
        tools: list[ToolDef],
        effort: Optional[str],     # None/'off' disables reasoning; else low..max
        max_tokens: int,
    ) -> AsyncIterator[Event]:
        ...
