"""Headless copilot benchmarking (studio CLI around dsl-eval-core + metagen-domain).

The scorers + task suite now live in metagen-domain; re-exported here for
backward-compatible imports.
"""
from metagen_domain import STARTER, load_suite, make_scorer

__all__ = ["make_scorer", "load_suite", "STARTER"]
