"""Build a Provider from already-resolved settings.

Config-shape-agnostic on purpose: the host (studio config, the benchmark
runner, a CAD plugin) resolves env-var keys / base_urls / per-model profiles
however it likes and hands this factory plain values. Keeps the copilot
package free of any studio/config coupling (see docs/COPILOT_PROVIDERS.md §9).
"""
from __future__ import annotations

from typing import Optional

from .anthropic import AnthropicProvider
from .base import Provider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

# aliases → canonical kind
_ALIASES = {
    "anthropic": "anthropic", "claude": "anthropic",
    "openai": "openai", "gpt": "openai", "responses": "openai",
    "vllm": "vllm", "openai-compat": "vllm", "openai_compat": "vllm",
    "chat_completions": "vllm",
    "gemini": "gemini", "google": "gemini",
}


def build_provider(kind: str, *, api_key: Optional[str] = None,
                   base_url: Optional[str] = None, mode: Optional[str] = None,
                   profile: Optional[dict] = None) -> Provider:
    canon = _ALIASES.get((kind or "anthropic").lower())
    if canon == "anthropic":
        return AnthropicProvider(api_key=api_key)
    if canon == "openai":
        return OpenAIProvider(api_key=api_key, base_url=base_url,
                              mode=mode or "responses", profile=profile)
    if canon == "vllm":
        return OpenAIProvider(api_key=api_key, base_url=base_url,
                              mode=mode or "chat_completions", profile=profile)
    if canon == "gemini":
        return GeminiProvider(api_key=api_key)
    raise ValueError(f"unknown provider kind: {kind!r}")
