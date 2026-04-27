import { useEffect, useRef, useState } from 'react';
import { streamChat } from '../api';
import type {
  ChatMessage, ChatStateContext, AssistantBlock, PendingProposal,
} from '../types';

interface Props {
  state: ChatStateContext;
  available: boolean;
  onApplyProposal: (newCode: string, proposalId: string, summary: string) => void;
  onGeometryDone: (summary: any) => void;
  onSimDone: (summary: any) => void;
}

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  blocks?: AssistantBlock[];        // for assistant
  text?: string;                    // for user (raw text)
  proposals?: PendingProposal[];    // proposals attached to this assistant turn
  toolResults?: { tool_id: string; name: string; result: any }[];
  streaming?: boolean;
}

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function ChatPanel(props: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const proposalsRef = useRef<Map<string, string>>(new Map()); // proposalId → toolUseId for status updates
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  function buildApiMessages(): ChatMessage[] {
    const out: ChatMessage[] = [];
    for (const t of turns) {
      if (t.role === 'user') {
        if (t.text) out.push({ role: 'user', content: t.text });
      } else {
        const blocks = t.blocks ?? [];
        if (blocks.length > 0) out.push({ role: 'assistant', content: blocks });
      }
    }
    return out;
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setError(null);
    setInput('');
    const userTurn: ChatTurn = { id: makeId(), role: 'user', text, blocks: [] };
    const assistantTurn: ChatTurn = {
      id: makeId(), role: 'assistant', blocks: [], streaming: true,
      proposals: [], toolResults: [],
    };
    setTurns((t) => [...t, userTurn, assistantTurn]);

    const ctl = new AbortController();
    abortRef.current = ctl;
    setBusy(true);
    try {
      const apiMessages: ChatMessage[] = [
        ...buildApiMessages(),
        { role: 'user', content: text },
      ];
      let liveText = '';
      for await (const ev of streamChat(apiMessages, props.state, ctl.signal)) {
        if (ev.kind === 'text') {
          liveText += ev.text;
          setTurns((t) => {
            const next = [...t];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              const blocks = [...(last.blocks ?? [])];
              const lastBlock = blocks[blocks.length - 1];
              if (lastBlock?.type === 'text') {
                blocks[blocks.length - 1] = { type: 'text', text: liveText };
              } else {
                blocks.push({ type: 'text', text: liveText });
              }
              next[next.length - 1] = { ...last, blocks };
            }
            return next;
          });
        } else if (ev.kind === 'tool_ui') {
          if (ev.payload?.kind === 'proposal') {
            const proposalId = makeId();
            proposalsRef.current.set(proposalId, ev.tool_id);
            setTurns((t) => {
              const next = [...t];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') {
                const proposals = [
                  ...(last.proposals ?? []),
                  {
                    id: proposalId, new_code: ev.payload.new_code,
                    summary: ev.payload.summary, status: 'pending' as const,
                  },
                ];
                next[next.length - 1] = { ...last, proposals };
              }
              return next;
            });
          } else if (ev.payload?.kind === 'geometry_done') {
            props.onGeometryDone(ev.payload);
          } else if (ev.payload?.kind === 'sim_done') {
            props.onSimDone(ev.payload);
          }
        } else if (ev.kind === 'tool_result') {
          setTurns((t) => {
            const next = [...t];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              const toolResults = [
                ...(last.toolResults ?? []),
                { tool_id: ev.tool_id, name: ev.name, result: ev.result },
              ];
              next[next.length - 1] = { ...last, toolResults };
            }
            return next;
          });
        } else if (ev.kind === 'assistant_msg') {
          // Reset live text accumulator for the *next* assistant turn (after a tool round-trip).
          liveText = '';
          setTurns((t) => {
            const next = [...t];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, blocks: ev.content };
            }
            return next;
          });
        } else if (ev.kind === 'done') {
          setTurns((t) => {
            const next = [...t];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, streaming: false };
            }
            return next;
          });
        } else if (ev.kind === 'error') {
          setError(ev.message);
          break;
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setError(e.message ?? String(e));
    } finally {
      setBusy(false);
      abortRef.current = null;
      setTurns((t) => {
        const next = [...t];
        const last = next[next.length - 1];
        if (last?.role === 'assistant') {
          next[next.length - 1] = { ...last, streaming: false };
        }
        return next;
      });
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  function applyProposal(turnId: string, proposal: PendingProposal) {
    props.onApplyProposal(proposal.new_code, proposal.id, proposal.summary);
    setTurns((t) =>
      t.map((tt) =>
        tt.id !== turnId
          ? tt
          : {
              ...tt,
              proposals: (tt.proposals ?? []).map((p) =>
                p.id === proposal.id ? { ...p, status: 'applied' } : p,
              ),
            },
      ),
    );
  }

  function discardProposal(turnId: string, proposal: PendingProposal) {
    setTurns((t) =>
      t.map((tt) =>
        tt.id !== turnId
          ? tt
          : {
              ...tt,
              proposals: (tt.proposals ?? []).map((p) =>
                p.id === proposal.id ? { ...p, status: 'discarded' } : p,
              ),
            },
      ),
    );
  }

  if (!props.available) {
    return (
      <div className="chat-disabled">
        <p><strong>Chat disabled.</strong></p>
        <p>Set <code>METAGEN_ANTHROPIC_API_KEY</code> in the backend environment and restart.</p>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-messages" ref={scrollRef}>
        {turns.length === 0 && (
          <div className="chat-empty">
            Ask the copilot to refactor your code, run a sim, or explain something.
            Edits come back as proposals you can accept or discard.
          </div>
        )}
        {turns.map((t) => (
          <Turn
            key={t.id}
            turn={t}
            onApply={(p) => applyProposal(t.id, p)}
            onDiscard={(p) => discardProposal(t.id, p)}
          />
        ))}
        {error && <div className="chat-error">error: {error}</div>}
      </div>
      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="message…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={busy}
        />
        <div className="chat-actions">
          {busy ? (
            <button onClick={cancel}>cancel</button>
          ) : (
            <button onClick={send} disabled={!input.trim()}>send</button>
          )}
        </div>
      </div>
    </div>
  );
}

interface TurnProps {
  turn: ChatTurn;
  onApply: (p: PendingProposal) => void;
  onDiscard: (p: PendingProposal) => void;
}

function Turn({ turn, onApply, onDiscard }: TurnProps) {
  if (turn.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-role">you</div>
        <div className="msg-text">{turn.text}</div>
      </div>
    );
  }
  return (
    <div className="msg msg-assistant">
      <div className="msg-role">copilot {turn.streaming && <span className="streaming">…</span>}</div>
      {(turn.blocks ?? []).map((b, i) => {
        if (b.type === 'text') {
          return <div key={i} className="msg-text">{b.text}</div>;
        }
        if (b.type === 'tool_use') {
          if (b.name === 'propose_edit') return null; // shown via proposals card
          return (
            <div key={i} className="tool-call">
              <span className="tool-name">{b.name}</span>
              <span className="tool-args">{JSON.stringify(b.input)}</span>
            </div>
          );
        }
        return null;
      })}
      {turn.proposals?.map((p) => (
        <div key={p.id} className={`proposal proposal-${p.status}`}>
          <div className="proposal-header">
            <span className="proposal-tag">proposed edit</span>
            <span className="proposal-summary">{p.summary}</span>
          </div>
          {p.status === 'pending' ? (
            <>
              <details>
                <summary>preview ({p.new_code.split('\n').length} lines)</summary>
                <pre>{p.new_code}</pre>
              </details>
              <div className="proposal-actions">
                <button onClick={() => onApply(p)}>apply</button>
                <button onClick={() => onDiscard(p)}>discard</button>
              </div>
            </>
          ) : (
            <div className="proposal-status">{p.status}</div>
          )}
        </div>
      ))}
      {turn.toolResults?.map((tr, i) => {
        if (tr.name === 'propose_edit') return null;
        return (
          <div key={i} className="tool-result">
            <span className="tool-name">{tr.name}</span>
            {tr.result?.ok === false ? (
              <span className="tool-err"> failed: {tr.result.error}</span>
            ) : tr.name === 'run_geometry' ? (
              <span> · vf {tr.result.volume_fraction?.toFixed(3)} · {tr.result.elapsed_s?.toFixed(2)}s</span>
            ) : tr.name === 'run_simulation' ? (
              <span> · E_VRH {tr.result.properties?.E_VRH?.toExponential(2)} · {tr.result.backend_used} · {tr.result.elapsed_s?.toFixed(2)}s</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
