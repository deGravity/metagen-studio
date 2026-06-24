"""metagen copilot — provider-neutral agent engine.

Self-contained: imports no studio/FastAPI internals (only injected interfaces),
so it can later move to its own package. See docs/COPILOT_PROVIDERS.md.
"""
from .benchmark import BenchmarkRunner, RunRecord, Solver, Task, aggregate
from .domain import Domain, registry_from_domain
from .engine import CopilotEngine
from .tools import Tool, ToolEnv, ToolOutcome, ToolRegistry
from . import types

__all__ = ["CopilotEngine", "Tool", "ToolEnv", "ToolOutcome", "ToolRegistry",
           "BenchmarkRunner", "Task", "RunRecord", "Solver", "aggregate",
           "Domain", "registry_from_domain", "types"]
