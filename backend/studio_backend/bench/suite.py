"""Starter benchmark task suite + a JSON loader.

A suite is a list of copilot.Task. The built-in starter set adapts the legacy
eval categories (material_understanding / inverse_design / reconstruction) to
the agentic engine. Load your own with `load_suite(path)` — a JSON list of
{id, prompt, category?, target?, initial_code?}.
"""
from __future__ import annotations

import json
from typing import Optional

from metagen_copilot import Task

STARTER: list[Task] = [
    Task(id="gyroid-basic",
         prompt="Write make_structure() for a simple gyroid TPMS unit cell with "
                "a moderate wall thickness. Propose the full program.",
         category="open"),
    Task(id="vf-030",
         prompt="Design a unit cell whose solid volume fraction is close to 0.30. "
                "Propose the full make_structure() program.",
         category="inverse_design",
         target={"vf": 0.30, "resolution": 33}),
    Task(id="vf-050-bcc",
         prompt="Design a BCC-style strut lattice with a volume fraction near 0.50. "
                "Propose the full program.",
         category="inverse_design",
         target={"vf": 0.50, "resolution": 33}),
    Task(id="explain-gyroid",
         prompt="In 3-4 sentences, what is a gyroid and why is it useful as a "
                "metamaterial unit cell?",
         category="material_understanding"),
]


def load_suite(path: Optional[str]) -> list[Task]:
    if not path:
        return list(STARTER)
    with open(path) as f:
        raw = json.load(f)
    return [Task(id=t["id"], prompt=t["prompt"],
                 category=t.get("category", "open"),
                 target=t.get("target", {}),
                 initial_code=t.get("initial_code", "")) for t in raw]
