"""Tool registry — decoupled from transport and from any specific host.

A Tool is a provider-neutral schema + an async handler. The handler receives
the model's args and a ToolEnv (the *environment* the host supplies — kernel
runner, compiled program, session id, …) and returns a ToolOutcome: a lean
result for the model plus optional sidecar Artifact events for consumers.

The studio binds run_geometry/run_simulation/propose_edit to kernel_job + UI
artifacts; a benchmark runner binds them to a headless env; a CAD host binds
its own. Same engine, different ToolEnv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .types import Artifact, ToolDef


@dataclass
class ToolEnv:
    """Opaque, host-provided execution context. The engine never inspects it;
    it just threads it to handlers. Concrete hosts subclass / fill it."""
    data: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return self.data.get(key, default)


@dataclass
class ToolOutcome:
    result: Any                          # JSON-able result returned to the model
    artifacts: list[Artifact] = field(default_factory=list)
    error: bool = False


ToolHandler = Callable[[dict, ToolEnv], Awaitable[ToolOutcome]]


@dataclass
class Tool:
    defn: ToolDef
    handler: ToolHandler


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.defn.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def defs(self) -> list[ToolDef]:
        return [t.defn for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
