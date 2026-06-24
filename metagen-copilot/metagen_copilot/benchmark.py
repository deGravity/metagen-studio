"""Headless benchmark runner — drive the CopilotEngine over a task suite.

The payoff of the provider/UI decoupling: the *same* engine + tools that power
the live studio, run with no browser, across any provider, then scored. This
module is host-agnostic (no studio/FastAPI/kernel imports) — the host injects:
  * an env_factory(Task) -> ToolEnv binding the tools to its geometry engine
    (the studio binds kernel_job; a CAD host binds its own),
  * an optional async scorer(Task, RunRecord) -> dict (kernel-backed in studio),
  * an optional log callback (the studio records each run as a session).
See docs/COPILOT_PROVIDERS.md §5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable

from .engine import CopilotEngine, LogFn
from .tools import ToolEnv
from .types import (
    AssistantMessage, Artifact, Done, ErrorEvent, Msg, Part, SystemBlock, Text,
    TextDelta, ThinkingDelta, ToolCall, ToolResultEvent, part_to_dict,
)


@dataclass
class Task:
    """One benchmark item. `category` drives the host's scorer; `target` holds
    scorer-specific expectations (target moduli / vf / voxel grid ref)."""
    id: str
    prompt: str
    category: str = "open"        # material_understanding|inverse_design|reconstruction|open
    attachments: list[Part] = field(default_factory=list)
    target: dict = field(default_factory=dict)
    initial_code: str = ""


@dataclass
class RunRecord:
    task_id: str
    model: str
    effort: Optional[str]
    repeat: int
    final_code: Optional[str] = None       # last propose_edit's new_code
    answer_text: str = ""                  # final assistant turn text
    n_turns: int = 0                       # model calls
    n_tool_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)   # [{name, input}]
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    elapsed_s: float = 0.0
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    events: list[dict] = field(default_factory=list)       # serialized transcript
    score: Optional[dict] = None           # filled by the injected scorer

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "task_id", "model", "effort", "repeat", "n_turns", "n_tool_calls",
            "input_tokens", "output_tokens", "thinking_tokens", "elapsed_s",
            "stop_reason", "error")}
        d["has_code"] = self.final_code is not None
        d["score"] = self.score
        return d


EnvFactory = Callable[[Task], ToolEnv]
Scorer = Callable[[Task, RunRecord], Awaitable[dict]]


@runtime_checkable
class Solver(Protocol):
    """A thing under evaluation: given a task + env, produce a RunRecord. The
    eval harness scores any Solver — our copilot (CopilotSolver, in the copilot
    package) is one; a non-copilot baseline is another. Eval depends on this
    protocol, never on the copilot. See ../../docs/repository_architecture.md §2."""

    async def solve(self, task: Task, env: ToolEnv) -> RunRecord:
        ...


class BenchmarkRunner:
    def __init__(self, engine: CopilotEngine, system: list[SystemBlock],
                 env_factory: EnvFactory, *,
                 scorer: Optional[Scorer] = None,
                 monotonic: Callable[[], float] = time.perf_counter):
        self.engine = engine
        self.system = system
        self.env_factory = env_factory
        self.scorer = scorer
        self._clock = monotonic

    async def run_task(self, task: Task, *, model: str,
                       effort: Optional[str] = None, max_tokens: int = 4096,
                       repeat: int = 0, log: Optional[LogFn] = None) -> RunRecord:
        rec = RunRecord(task_id=task.id, model=model, effort=effort, repeat=repeat,
                        final_code=task.initial_code or None)
        messages = [Msg("user", [Text(task.prompt), *task.attachments])]
        env = self.env_factory(task)
        t0 = self._clock()
        cur_text: list[str] = []
        try:
            async for ev in self.engine.run(model=model, system=self.system,
                                            messages=messages, env=env,
                                            effort=effort, max_tokens=max_tokens,
                                            log=log):
                if isinstance(ev, TextDelta):
                    cur_text.append(ev.text)
                elif isinstance(ev, ThinkingDelta):
                    pass
                elif isinstance(ev, AssistantMessage):
                    rec.n_turns += 1
                    if ev.usage:
                        rec.input_tokens += ev.usage.input_tokens or 0
                        rec.output_tokens += ev.usage.output_tokens or 0
                        rec.thinking_tokens += ev.usage.thinking_tokens or 0
                    txt = "".join(p.text for p in ev.parts if isinstance(p, Text))
                    if txt:
                        rec.answer_text = txt        # last turn's text wins
                    for p in ev.parts:
                        if isinstance(p, ToolCall):
                            rec.tool_calls.append({"name": p.name, "input": p.input})
                    cur_text = []
                elif isinstance(ev, Artifact):
                    if ev.kind == "proposal":
                        code = ev.data.get("new_code")
                        if code is not None:
                            rec.final_code = code
                elif isinstance(ev, ToolResultEvent):
                    rec.n_tool_calls += 1
                    rec.events.append({"type": "tool_result", "name": ev.name,
                                       "result": ev.result, "elapsed_s": ev.elapsed_s})
                elif isinstance(ev, Done):
                    rec.stop_reason = ev.stop_reason
                elif isinstance(ev, ErrorEvent):
                    rec.error = ev.message
        except Exception as exc:  # noqa: BLE001 — a task failure must not abort the sweep
            rec.error = f"{type(exc).__name__}: {exc}"
        rec.elapsed_s = round(self._clock() - t0, 3)

        if self.scorer is not None and rec.error is None:
            try:
                rec.score = await self.scorer(task, rec)
            except Exception as exc:  # noqa: BLE001
                rec.score = {"error": f"{type(exc).__name__}: {exc}"}
        return rec

    async def run_suite(self, tasks: list[Task], models: list[str], *,
                        effort: Optional[str] = None, max_tokens: int = 4096,
                        repeats: int = 1,
                        log_factory: Optional[Callable[[Task, str, int], LogFn]] = None,
                        ) -> list[RunRecord]:
        """Sweep every (model, task, repeat). Sequential by design — the kernel
        is the bottleneck and concurrent solves contend; the caller can shard."""
        out: list[RunRecord] = []
        for model in models:
            for task in tasks:
                for r in range(max(1, repeats)):
                    log = log_factory(task, model, r) if log_factory else None
                    out.append(await self.run_task(task, model=model, effort=effort,
                                                   max_tokens=max_tokens, repeat=r,
                                                   log=log))
        return out


def aggregate(records: list[RunRecord]) -> list[dict]:
    """Collapse repeats into one row per (model, task): means + score rollup."""
    from statistics import mean
    groups: dict[tuple, list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.model, r.task_id), []).append(r)
    rows = []
    for (model, task_id), recs in groups.items():
        ok = [r for r in recs if r.error is None]
        scores = [r.score.get("score") for r in recs
                  if r.score and isinstance(r.score.get("score"), (int, float))]
        rows.append({
            "model": model, "task": task_id, "n": len(recs),
            "ok": len(ok),
            "mean_turns": round(mean([r.n_turns for r in recs]), 2) if recs else 0,
            "mean_tool_calls": round(mean([r.n_tool_calls for r in recs]), 2) if recs else 0,
            "mean_out_tokens": round(mean([r.output_tokens for r in recs]), 0) if recs else 0,
            "mean_think_tokens": round(mean([r.thinking_tokens for r in recs]), 0) if recs else 0,
            "mean_elapsed_s": round(mean([r.elapsed_s for r in recs]), 2) if recs else 0,
            "mean_score": round(mean(scores), 4) if scores else None,
        })
    return rows
