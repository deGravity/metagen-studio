"""Kernel-backed scorers for the headless benchmark (studio wiring).

Each scorer turns a finished RunRecord into {score: float in [0,1], ...details}.
Geometry/sim run through the same subprocess kernel runners the studio uses.
Metric formulas follow the legacy eval suite (IoU on voxels, abs-error on
moduli). Dispatched by Task.category. See docs/COPILOT_PROVIDERS.md §5.
"""
from __future__ import annotations

import base64
from typing import Optional

import numpy as np

from dsl_eval_core import RunRecord, Task
from ..kernel_job import run_geometry_result, run_sim_result
from ..state import program_cache


def _rel(actual: float, target: float) -> float:
    """1.0 when actual==target, decaying to 0 as the relative error → 1."""
    if target == 0:
        return 1.0 if actual == 0 else 0.0
    return max(0.0, 1.0 - abs(actual - target) / abs(target))


async def _geometry(code: str, resolution: int, k: int) -> dict:
    return await run_geometry_result(code, resolution, k)


def _unpack_voxels(b64: str, shape: tuple) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return np.unpackbits(raw)[: int(np.prod(shape))].astype(bool).reshape(shape)


async def score_open(task: Task, rec: RunRecord, *, resolution: int) -> dict:
    """No target: reward a program that compiles and produces non-empty
    geometry. material_understanding (free-text Q&A) is graded only on whether
    an answer was produced — real grading needs an LLM judge (future)."""
    if task.category == "material_understanding":
        return {"score": 1.0 if rec.answer_text.strip() else 0.0,
                "answered": bool(rec.answer_text.strip()),
                "note": "free-text; not auto-graded (LLM-judge TODO)"}
    if not rec.final_code:
        return {"score": 0.0, "reason": "no code proposed"}
    compiled = program_cache.get_or_compile(rec.final_code)
    if compiled.error:
        return {"score": 0.0, "compiles": False, "error": compiled.error}
    try:
        g = await _geometry(rec.final_code, resolution, 1)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "compiles": True, "ran": False, "error": str(exc)}
    nonempty = g["n_active_voxels"] > 0
    return {"score": 1.0 if nonempty else 0.0, "compiles": True, "ran": True,
            "volume_fraction": g["volume_fraction"], "nonempty": nonempty}


async def score_inverse_design(task: Task, rec: RunRecord, *, resolution: int) -> dict:
    """target: {property: name (default E_VRH), value, [vf], [E], [nu], [tol]}.
    Runs sim on the final code; score = closeness of the target property (and
    vf if given) to the requested value."""
    if not rec.final_code:
        return {"score": 0.0, "reason": "no code proposed"}
    t = task.target
    res = int(t.get("resolution", resolution))
    try:
        s = await run_sim_result(rec.final_code, res, 1, t.get("backend", "auto"),
                                 float(t.get("E", 1.0)), float(t.get("nu", 0.45)))
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "ran": False, "error": str(exc)}
    props = s.get("properties", {})
    parts: list[float] = []
    detail: dict = {"backend_used": s.get("solver_used")}
    if "value" in t:
        prop = t.get("property", "E_VRH")
        got = float(props.get(prop, 0.0))
        parts.append(_rel(got, float(t["value"])))
        detail[prop] = got
        detail[f"{prop}_target"] = t["value"]
    if "vf" in t:
        g = await _geometry(rec.final_code, res, 1)
        parts.append(_rel(g["volume_fraction"], float(t["vf"])))
        detail["volume_fraction"] = g["volume_fraction"]
        detail["vf_target"] = t["vf"]
    detail["score"] = round(sum(parts) / len(parts), 4) if parts else 0.0
    return detail


async def score_reconstruction(task: Task, rec: RunRecord, *, resolution: int) -> dict:
    """target: {voxels_npz: path, [key]} — IoU of the final code's voxels vs a
    reference active-cell grid (legacy packed-bool .npz)."""
    if not rec.final_code:
        return {"score": 0.0, "reason": "no code proposed"}
    t = task.target
    ref_path = t.get("voxels_npz")
    if not ref_path:
        return {"score": 0.0, "reason": "no reference voxels in target"}
    res = int(t.get("resolution", resolution))
    try:
        npz = np.load(ref_path)
        ref = npz[t.get("key", "vox_active_cells")].astype(bool)
        g = await _geometry(rec.final_code, res, 1)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "error": str(exc)}
    got = _unpack_voxels(g["voxels_b64"], ref.shape) if "voxels_b64" in g else None
    if got is None or got.shape != ref.shape:
        return {"score": 0.0, "reason": "voxel shape mismatch / unavailable",
                "ref_shape": list(ref.shape)}
    inter = (got & ref).sum()
    union = (got | ref).sum()
    iou = float(inter / union) if union else 0.0
    return {"score": round(iou, 4), "iou": round(iou, 4)}


def make_scorer(resolution: int = 33):
    """Return an async scorer(task, rec) dispatching on task.category."""
    async def scorer(task: Task, rec: RunRecord) -> dict:
        if task.category == "inverse_design":
            return await score_inverse_design(task, rec, resolution=resolution)
        if task.category == "reconstruction":
            return await score_reconstruction(task, rec, resolution=resolution)
        return await score_open(task, rec, resolution=resolution)
    return scorer
