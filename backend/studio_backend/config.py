"""Layered configuration for metagen-studio.

Resolution order (lowest → highest precedence), deep-merged:
  1. packaged default  metagen-studio/config.yaml
  2. ~/.config/metagen.yaml
  3. ~/.metagenconfig.yaml
  4. environment variables (per-key, see _ENV_OVERRIDES)

Access via get_config() (cached) or cfg(path, default) for dotted lookups,
e.g. cfg("sessions.dir") or cfg("copilot.thinking.budget_tokens", 4000).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# packaged default lives at metagen-studio/config.yaml (two dirs up from this
# file: studio_backend/ -> backend/ -> metagen-studio/)
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config.yaml'

# Override files, lowest → highest precedence.
_OVERRIDE_FILES = [
    Path.home() / '.config' / 'metagen.yaml',
    Path.home() / '.metagenconfig.yaml',
]

# env var -> dotted config path. String values; path-like ones get ~-expanded.
_ENV_OVERRIDES = {
    'METAGEN_STUDIO_SESSION_DIR': 'sessions.dir',
    'METAGEN_STUDIO_SESSIONS_ENABLED': 'sessions.enabled',
    'METAGEN_STUDIO_THINKING_ENABLED': 'copilot.thinking.enabled',
    'METAGEN_STUDIO_THINKING_EFFORT': 'copilot.thinking.effort',
    'METAGEN_STUDIO_AUTONAME_ENABLED': 'copilot.autoname.enabled',
    'METAGEN_STUDIO_AUTONAME_MODEL': 'copilot.autoname.model',
}

_PATH_KEYS = {'sessions.dir'}          # values that should be ~-expanded
_BOOL_KEYS = {'sessions.enabled', 'copilot.thinking.enabled',
              'copilot.autoname.enabled'}
_INT_KEYS = {'copilot.autoname.every_turns', 'copilot.autoname.idle_seconds'}

_cache: dict | None = None


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001 — a malformed override shouldn't crash startup
        return {}


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            return
    cur[keys[-1]] = value


def _coerce(dotted: str, raw: str) -> Any:
    if dotted in _BOOL_KEYS:
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    if dotted in _INT_KEYS:
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def _build() -> dict:
    cfg = _load_yaml(_DEFAULT_CONFIG)
    for f in _OVERRIDE_FILES:
        cfg = _deep_merge(cfg, _load_yaml(f))
    for env, dotted in _ENV_OVERRIDES.items():
        if env in os.environ:
            _set_dotted(cfg, dotted, _coerce(dotted, os.environ[env]))
    # normalize path-like keys
    for dotted in _PATH_KEYS:
        val = cfg_from(cfg, dotted)
        if isinstance(val, str):
            _set_dotted(cfg, dotted, str(Path(val).expanduser()))
    return cfg


def cfg_from(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for k in dotted.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_config() -> dict:
    global _cache
    if _cache is None:
        _cache = _build()
    return _cache


def cfg(dotted: str, default: Any = None) -> Any:
    return cfg_from(get_config(), dotted, default)


def session_dir() -> Path:
    """Resolved, ~-expanded sessions directory (created on demand by callers)."""
    return Path(cfg('sessions.dir', str(Path.home() / '.metagen-studio' / 'sessions')))


def reload_config() -> dict:
    """Force a re-read (e.g. after editing a config file in dev)."""
    global _cache
    _cache = None
    return get_config()
