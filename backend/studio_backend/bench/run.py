"""Benchmark CLI — sweep provider×model×task through the headless engine.

Reuses the studio's own registry + system prompt so we measure the *real*
copilot, just with no UI. Each run can be recorded as a session (events.jsonl +
DAG) for inspection in the Log Explorer.

    python -m studio_backend.bench.run \
        --models claude-opus-4-7,Qwen/Qwen3.6-35B-A3B-FP8 \
        --provider-for "Qwen/Qwen3.6-35B-A3B-FP8=vllm" \
        --effort medium --repeats 1 --resolution 33 \
        --out bench_results.json --record

Provider is inferred from the model name unless overridden via --provider (one
provider for all models) or --provider-for model=provider (per model).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from typing import Optional

from .. import chat as _chat
from .. import sessions as _sess
from ..copilot import BenchmarkRunner, CopilotEngine, ToolEnv, aggregate
from ..copilot.types import SystemBlock
from ..models import ChatRequest, ChatMessage, ChatStateContext
from .scoring import make_scorer
from .suite import load_suite


def _provider_for(model: str, default_provider: Optional[str],
                  per_model: dict[str, str]):
    """Resolve a provider for `model` via the studio's own resolver (handles
    key/base_url gating). Returns (provider, error). When no provider is given
    we infer from the model name rather than fall back to the global config
    default — a sweep spans models, so the single config default is unhelpful."""
    name = per_model.get(model) or default_provider or _chat._infer_provider(model)
    req = ChatRequest(messages=[ChatMessage(role="user", content="")],
                      state=ChatStateContext(code=""), model=model, provider=name)
    return _chat._resolve_provider(req)


def _parse_provider_for(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" in it:
            m, p = it.split("=", 1)
            out[m.strip()] = p.strip()
    return out


def _session_log_factory(model: str):
    """Per-run: create a session, return (log, finalize). finalize() makes the
    assistant_turn node so the run shows up in the Log Explorer."""
    def factory(task, mdl, repeat):
        try:
            tree = _sess.create_session(name=f"bench:{mdl}:{task.id}#{repeat}", model=mdl)
            sid = tree["session_id"]
        except Exception:  # noqa: BLE001
            return None
        ev_ids: list[str] = []
        _sess.append_event(sid, "user_message", {"content": task.prompt})

        def log(etype, payload):
            try:
                ev_ids.append(_sess.append_event(sid, etype, payload)["id"])
            except Exception:  # noqa: BLE001
                pass
        log._sid = sid          # type: ignore[attr-defined]
        log._evids = ev_ids     # type: ignore[attr-defined]
        return log
    return factory


async def _amain(args) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model = _parse_provider_for(args.provider_for)
    tasks = load_suite(args.suite)
    system = [SystemBlock(_chat._static_system_text(), cache=True)]
    scorer = make_scorer(resolution=args.resolution)
    registry = _chat._build_registry()
    log_factory = _session_log_factory(args.models) if args.record else None

    all_records = []
    for model in models:
        provider, perr = _provider_for(model, args.provider, per_model)
        if perr:
            print(f"[skip] {model}: {perr}")
            continue
        engine = CopilotEngine(provider, registry, max_turns=args.max_turns)
        runner = BenchmarkRunner(
            engine, system,
            env_factory=lambda t: ToolEnv(data={"state": ChatStateContext(code=t.initial_code)}),
            scorer=scorer)
        for task in tasks:
            for r in range(max(1, args.repeats)):
                log = log_factory(task, model, r) if log_factory else None
                print(f"[run] {model} · {task.id} · rep{r} …", flush=True)
                rec = await runner.run_task(task, model=model, effort=args.effort,
                                            max_tokens=args.max_tokens, repeat=r, log=log)
                if log is not None:
                    try:
                        _sess.add_node(log._sid, "assistant_turn",  # type: ignore[attr-defined]
                                       f"bench {task.id}",
                                       {"code": rec.final_code, "code_hash": None,
                                        "geometry_ref": None, "sim_ref": None, "chat_len": 1},
                                       event_ids=list(log._evids))   # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                print(f"      turns={rec.n_turns} tools={rec.n_tool_calls} "
                      f"out_tok={rec.output_tokens} {rec.elapsed_s}s "
                      f"score={rec.score.get('score') if rec.score else None} "
                      f"{'ERR: ' + rec.error if rec.error else ''}")
                all_records.append(rec)

    print("\n=== summary (per model×task, mean over repeats) ===")
    rows = aggregate(all_records)
    if rows:
        cols = ["model", "task", "n", "ok", "mean_turns", "mean_tool_calls",
                "mean_out_tokens", "mean_think_tokens", "mean_elapsed_s", "mean_score"]
        print(" | ".join(cols))
        for row in rows:
            print(" | ".join(str(row.get(c)) for c in cols))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"records": [dataclasses.asdict(r) for r in all_records],
                       "summary": rows}, f, indent=2)
        print(f"\nwrote {len(all_records)} records → {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless copilot benchmark sweep.")
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--provider", default=None, help="force one provider for all models")
    ap.add_argument("--provider-for", action="append", default=[],
                    help="per-model override, e.g. 'Qwen/Qwen3.6=vllm' (repeatable)")
    ap.add_argument("--effort", default=None, help="off|low|medium|high|xhigh|max")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=33)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--suite", default=None, help="JSON suite path (default: built-in starter)")
    ap.add_argument("--out", default=None, help="write full records JSON here")
    ap.add_argument("--record", action="store_true", help="record each run as a session")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
