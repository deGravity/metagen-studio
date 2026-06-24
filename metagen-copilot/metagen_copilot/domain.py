"""Domain registration seam.

A `Domain` is a domain pack's contribution to the generic machinery: tools (for
the copilot), scorers (for the eval harness), and prompt/docs/ui. The machinery
consumes a `Domain` without importing the domain's runtime — the runtime is
touched only inside the domain-supplied tool handlers and scorers. This is the
plug-in point that lets sister projects reuse the engine/harness/shell.
See ../../docs/repository_architecture.md §3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .tools import Tool, ToolRegistry


@dataclass
class Domain:
    name: str
    tools: list[Tool] = field(default_factory=list)
    scorers: dict[str, Any] = field(default_factory=dict)   # category -> Scorer
    system_text: str = ""
    docs: str = ""
    ui: Optional[Any] = None


def registry_from_domain(domain: Domain) -> ToolRegistry:
    reg = ToolRegistry()
    for t in domain.tools:
        reg.register(t)
    return reg
