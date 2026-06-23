// LLM provider/model selection state, browser-side.
//
// Credentials entered in the advanced settings panel live ONLY in localStorage
// and are sent per chat request (the backend makes the API call). The per-
// session provider/model choice is also persisted in localStorage, keyed by
// session id, so switching sessions restores the choice.
import type { LlmCreds, ProviderInfo } from './types';

const CREDS_KEY = 'studio.llm.creds';
const SEL_PREFIX = 'studio.llm.sel.';            // + sessionId -> {provider, model}
const CUSTOM_MODELS_KEY = 'studio.llm.customModels';  // {provider: string[]}

export function loadCreds(): LlmCreds {
  try { return JSON.parse(localStorage.getItem(CREDS_KEY) || '{}'); } catch { return {}; }
}
export function saveCreds(c: LlmCreds): void {
  localStorage.setItem(CREDS_KEY, JSON.stringify(c));
}

export function loadCustomModels(): Record<string, string[]> {
  try { return JSON.parse(localStorage.getItem(CUSTOM_MODELS_KEY) || '{}'); } catch { return {}; }
}
export function saveCustomModels(m: Record<string, string[]>): void {
  localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(m));
}

export interface Selection { provider: string; model: string; }

export function loadSelection(sessionId: string | undefined): Selection | null {
  if (!sessionId) return null;
  try { return JSON.parse(localStorage.getItem(SEL_PREFIX + sessionId) || 'null'); }
  catch { return null; }
}
export function saveSelection(sessionId: string | undefined, sel: Selection): void {
  if (!sessionId) return;
  localStorage.setItem(SEL_PREFIX + sessionId, JSON.stringify(sel));
}

// A provider is usable if the backend has creds OR the browser supplied them.
export function isAvailable(p: ProviderInfo, creds: LlmCreds): boolean {
  if (p.available) return true;
  const c = creds[p.name] || {};
  return p.needs_base_url ? !!c.base_url : !!c.api_key;
}

// Models to offer for a provider. For discoverable providers (vLLM) the live
// list from the server leads; otherwise the curated list. Custom models the
// user added are always appended.
export function modelsFor(p: ProviderInfo, custom: Record<string, string[]>,
                          discovered?: Record<string, string[]>): string[] {
  const extra = custom[p.name] || [];
  const disc = (discovered && discovered[p.name]) || [];
  const base = p.discover ? disc : (p.models || []);
  return Array.from(new Set([...base, ...extra]));
}

// Effective base_url for a provider: client-supplied wins, else server-config.
export function baseUrlFor(p: ProviderInfo, creds: LlmCreds): string | undefined {
  return creds[p.name]?.base_url || p.base_url || undefined;
}

// The credential override to send with a request for `provider` (only when the
// backend isn't already configured for it — otherwise let the backend use its
// own env/config and send nothing).
export function credsToSend(provider: string, providers: ProviderInfo[], creds: LlmCreds):
    { api_key?: string; base_url?: string } {
  const p = providers.find((x) => x.name === provider);
  if (p && p.available) return {};       // backend already has it
  const c = creds[provider] || {};
  const out: { api_key?: string; base_url?: string } = {};
  if (c.api_key) out.api_key = c.api_key;
  if (c.base_url) out.base_url = c.base_url;
  return out;
}
