"""Shared singletons (program cache) so chat and HTTP route handlers
both see the same compiled-Structure store without circular imports.
"""
from __future__ import annotations

from .execute import ProgramCache

program_cache = ProgramCache(max_entries=32)
