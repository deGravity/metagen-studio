import { useState } from 'react';
import type { LlmCreds, ProviderInfo } from '../types';
import {
  Selection, isAvailable, modelsFor, saveCreds, saveCustomModels,
} from '../llm';

interface Props {
  providers: ProviderInfo[];
  creds: LlmCreds;
  setCreds: (c: LlmCreds) => void;
  customModels: Record<string, string[]>;
  setCustomModels: (m: Record<string, string[]>) => void;
  discovered?: Record<string, string[]>;
  discErr?: Record<string, string>;
  selection: Selection;
  onSelect: (s: Selection) => void;
  onRefresh?: (provider: string) => void;
}

export function LlmSelector(props: Props) {
  const { providers, creds, selection, onSelect } = props;
  const [showSettings, setShowSettings] = useState(false);

  const cur = providers.find((p) => p.name === selection.provider);
  const models = cur ? modelsFor(cur, props.customModels, props.discovered) : [];

  function pickProvider(name: string) {
    const p = providers.find((x) => x.name === name);
    const ms = p ? modelsFor(p, props.customModels, props.discovered) : [];
    // never inherit the previous provider's model (e.g. a leftover claude-* on
    // vLLM); fall back to '' so the dropdown shows a placeholder until a real
    // model for this provider is chosen / discovered.
    const model = ms.includes(selection.model) ? selection.model
      : (p?.default_model || ms[0] || '');
    onSelect({ provider: name, model });
  }

  const modelValid = models.includes(selection.model);

  return (
    <div className="llm-selector">
      <select className="llm-provider" value={selection.provider}
              onChange={(e) => pickProvider(e.target.value)} title="LLM provider">
        {providers.map((p) => {
          const ok = isAvailable(p, creds);
          return (
            <option key={p.name} value={p.name} disabled={!ok}>
              {p.label}{ok ? '' : ` (needs ${p.needs_base_url ? 'URL' : 'key'})`}
            </option>
          );
        })}
      </select>
      <select className="llm-model" value={modelValid ? selection.model : ''}
              onChange={(e) => onSelect({ ...selection, model: e.target.value })}
              title={cur?.discover ? (props.discErr?.[selection.provider] || 'discovered models') : 'model'}>
        {!modelValid && (
          <option value="" disabled>
            {models.length ? 'select a model'
              : (cur?.discover
                  ? (props.discErr?.[selection.provider]
                      ? `no models (${props.discErr[selection.provider]})` : 'discovering…')
                  : 'no models')}
          </option>
        )}
        {models.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
      {cur?.discover && props.onRefresh && (
        <button className="llm-gear" title="re-discover models from the server"
                onClick={() => props.onRefresh!(selection.provider)}>↻</button>
      )}
      <button className="llm-gear" title="LLM settings (keys, vLLM URL, custom models)"
              onClick={() => setShowSettings(true)}>⚙</button>
      {showSettings && (
        <LlmSettingsModal {...props} onClose={() => setShowSettings(false)} />
      )}
    </div>
  );
}

function LlmSettingsModal(props: Props & { onClose: () => void }) {
  const { providers, onClose } = props;
  // local working copies so edits apply on Save
  const [creds, setLocalCreds] = useState<LlmCreds>(() => JSON.parse(JSON.stringify(props.creds)));
  const [custom, setLocalCustom] = useState<Record<string, string[]>>(
    () => JSON.parse(JSON.stringify(props.customModels)));
  const [newModel, setNewModel] = useState<Record<string, string>>({});

  function setCred(name: string, field: 'api_key' | 'base_url', v: string) {
    setLocalCreds((c) => ({ ...c, [name]: { ...(c[name] || {}), [field]: v } }));
  }
  function addModel(name: string) {
    const m = (newModel[name] || '').trim();
    if (!m) return;
    setLocalCustom((c) => ({ ...c, [name]: Array.from(new Set([...(c[name] || []), m])) }));
    setNewModel((n) => ({ ...n, [name]: '' }));
  }
  function removeModel(name: string, m: string) {
    setLocalCustom((c) => ({ ...c, [name]: (c[name] || []).filter((x) => x !== m) }));
  }
  function save() {
    saveCreds(creds); props.setCreds(creds);
    saveCustomModels(custom); props.setCustomModels(custom);
    onClose();
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal llm-settings" onClick={(e) => e.stopPropagation()}>
        <h3>LLM settings</h3>
        <p className="llm-note">
          Keys and URLs are stored only in this browser (localStorage) and sent
          with each request so the backend can call the provider. They are not
          saved on the server or in session logs.
        </p>
        {providers.map((p) => (
          <div key={p.name} className="llm-prov-block">
            <div className="llm-prov-head">
              <b>{p.label}</b>
              <span className={p.available ? 'llm-ok' : 'llm-warn'}>
                {p.available ? 'configured on server' : `needs ${p.needs}`}
              </span>
            </div>
            {p.needs_base_url && (
              <label className="llm-field">base URL
                <input type="text" placeholder="http://host:8000/v1"
                       value={creds[p.name]?.base_url || ''}
                       onChange={(e) => setCred(p.name, 'base_url', e.target.value)} />
              </label>
            )}
            <label className="llm-field">API key{p.available ? ' (override)' : ''}
              <input type="password" placeholder={p.available ? '•••• (server has one)' : 'paste key'}
                     value={creds[p.name]?.api_key || ''}
                     onChange={(e) => setCred(p.name, 'api_key', e.target.value)} />
            </label>
            <div className="llm-models">
              <span className="llm-models-label">custom models:</span>
              {(custom[p.name] || []).map((m) => (
                <span key={m} className="llm-chip">{m}
                  <button onClick={() => removeModel(p.name, m)}>×</button>
                </span>
              ))}
              <input type="text" placeholder="add model id" value={newModel[p.name] || ''}
                     onChange={(e) => setNewModel((n) => ({ ...n, [p.name]: e.target.value }))}
                     onKeyDown={(e) => { if (e.key === 'Enter') addModel(p.name); }} />
            </div>
          </div>
        ))}
        <div className="modal-actions">
          <button onClick={onClose}>cancel</button>
          <button className="primary" onClick={save}>save</button>
        </div>
      </div>
    </div>
  );
}
