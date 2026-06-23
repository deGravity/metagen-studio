import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { streamChat, uploadChatFile, getTranscript } from '../api';
import type {
  ChatMessage, ChatStateContext, AssistantBlock, PendingProposal,
  Attachment, UserContentBlock,
} from '../types';

interface Props {
  state: ChatStateContext;
  available: boolean;
  sessionId?: string;
  thinking?: boolean;
  // Bumped by the host to request a chat rehydrate (session switch / restore /
  // checkout); restoreNode picks the conversation prefix (undefined = HEAD).
  restoreToken?: number;
  restoreNode?: string;
  onApplyProposal: (newCode: string, proposalId: string, summary: string) => void;
  onGeometryDone: (summary: any) => void;
  onSimDone: (summary: any) => void;
  onTurnDone?: () => void;
}

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  blocks?: AssistantBlock[];        // for assistant
  text?: string;                    // for user (raw text)
  attachments?: Attachment[];       // for user (uploaded files)
  proposals?: PendingProposal[];    // proposals attached to this assistant turn
  toolResults?: { tool_id: string; name: string; result: any }[];
  thinking?: string;                // accumulated extended-thinking text
  streaming?: boolean;
}

const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const PDF_TYPE = 'application/pdf';
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;       // 5 MB per Anthropic guidance
// PDFs go via the Files API (uploaded once, referenced by file_id), so
// the inline 32 MB cap no longer applies. The backend enforces its own
// sanity ceiling — see _UPLOAD_MAX_BYTES in chat.py.
const MAX_PDF_BYTES = 100 * 1024 * 1024;       // 100 MB

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function humanSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

async function readAsDataUrl(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error ?? new Error('read failed'));
    r.readAsDataURL(file);
  });
}

async function buildImageAttachment(file: File): Promise<Attachment> {
  const dataUrl = await readAsDataUrl(file);
  const comma = dataUrl.indexOf(',');
  const meta = dataUrl.slice(0, comma);          // "data:image/png;base64"
  const data = dataUrl.slice(comma + 1);
  const mediaType = meta.slice(5, meta.indexOf(';'));
  return {
    id: makeId(),
    kind: 'image',
    mediaType,
    filename: file.name || 'image',
    size: file.size,
    dataB64: data,
    previewUrl: dataUrl,
  };
}

function newPdfAttachment(file: File): Attachment {
  return {
    id: makeId(),
    kind: 'document',
    mediaType: 'application/pdf',
    filename: file.name || 'document.pdf',
    size: file.size,
    uploading: true,
  };
}

function validateFile(file: File): string | null {
  if (IMAGE_TYPES.includes(file.type)) {
    if (file.size > MAX_IMAGE_BYTES) {
      return `image "${file.name}" exceeds ${humanSize(MAX_IMAGE_BYTES)} limit`;
    }
    return null;
  }
  if (file.type === PDF_TYPE) {
    if (file.size > MAX_PDF_BYTES) {
      return `pdf "${file.name}" exceeds ${humanSize(MAX_PDF_BYTES)} limit`;
    }
    return null;
  }
  return `unsupported type "${file.type || file.name}"; png/jpeg/gif/webp/pdf only`;
}

function attachmentToBlock(a: Attachment): UserContentBlock {
  if (a.kind === 'image') {
    if (!a.dataB64) throw new Error(`image ${a.filename} missing data`);
    return {
      type: 'image',
      source: { type: 'base64', media_type: a.mediaType, data: a.dataB64 },
    };
  }
  if (!a.fileId) throw new Error(`pdf ${a.filename} not yet uploaded`);
  return {
    type: 'document',
    source: { type: 'file', file_id: a.fileId },
  };
}

function buildUserContent(text: string, atts: Attachment[]): string | UserContentBlock[] {
  if (atts.length === 0) return text;
  const blocks: UserContentBlock[] = atts.map(attachmentToBlock);
  if (text) blocks.push({ type: 'text', text });
  return blocks;
}

export function ChatPanel(props: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState('');
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const proposalsRef = useRef<Map<string, string>>(new Map()); // proposalId → toolUseId for status updates
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  // Rehydrate the transcript from the event log on session switch / restore /
  // checkout. The backend is the source of truth; we mirror it here so a page
  // reload or a rewind doesn't lose the conversation + attachments. Skipped
  // while streaming so we never clobber an in-flight turn.
  useEffect(() => {
    const sid = props.sessionId;
    if (!sid) { setTurns([]); return; }
    if (busy) return;
    let cancelled = false;
    getTranscript(sid, props.restoreNode)
      .then((t) => { if (!cancelled) setTurns(t as ChatTurn[]); })
      .catch(() => { /* leave current turns on failure */ });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.sessionId, props.restoreToken]);

  function buildApiMessages(): ChatMessage[] {
    const out: ChatMessage[] = [];
    for (const t of turns) {
      if (t.role === 'user') {
        const text = t.text ?? '';
        const atts = t.attachments ?? [];
        if (!text && atts.length === 0) continue;
        out.push({ role: 'user', content: buildUserContent(text, atts) });
      } else {
        const blocks = t.blocks ?? [];
        if (blocks.length > 0) out.push({ role: 'assistant', content: blocks });
      }
    }
    return out;
  }

  async function addFiles(files: FileList | File[]) {
    setError(null);
    const arr = Array.from(files);
    for (const f of arr) {
      const err = validateFile(f);
      if (err) { setError(err); continue; }
      if (f.type === PDF_TYPE) {
        const att = newPdfAttachment(f);
        setPendingAttachments((cur) => [...cur, att]);
        uploadChatFile(f)
          .then((resp) => {
            setPendingAttachments((cur) =>
              cur.map((a) => a.id === att.id
                ? { ...a, fileId: resp.file_id, uploading: false }
                : a,
              ),
            );
          })
          .catch((e: any) => {
            const msg = e?.message ?? String(e);
            setPendingAttachments((cur) =>
              cur.map((a) => a.id === att.id
                ? { ...a, uploading: false, uploadError: msg }
                : a,
              ),
            );
            setError(`upload "${f.name}": ${msg}`);
          });
      } else {
        try {
          const att = await buildImageAttachment(f);
          setPendingAttachments((cur) => [...cur, att]);
        } catch (e: any) {
          setError(`failed to read "${f.name}": ${e.message ?? e}`);
        }
      }
    }
  }

  const uploadsPending = pendingAttachments.some((a) => a.uploading);
  const uploadsFailed = pendingAttachments.some((a) => a.uploadError);

  function removeAttachment(id: string) {
    setPendingAttachments((cur) => cur.filter((a) => a.id !== id));
  }

  async function send() {
    const text = input.trim();
    const atts = pendingAttachments;
    if ((!text && atts.length === 0) || busy) return;
    if (atts.some((a) => a.uploading)) {
      setError('wait for attachments to finish uploading');
      return;
    }
    if (atts.some((a) => a.uploadError)) {
      setError('remove failed attachments before sending');
      return;
    }
    setError(null);
    setInput('');
    setPendingAttachments([]);
    const userTurn: ChatTurn = {
      id: makeId(), role: 'user', text, attachments: atts, blocks: [],
    };
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
        { role: 'user', content: buildUserContent(text, atts) },
      ];
      let liveText = '';
      let liveThinking = '';
      for await (const ev of streamChat(apiMessages, props.state, ctl.signal,
          'claude-opus-4-7', { thinking: props.thinking, session_id: props.sessionId })) {
        if (ev.kind === 'thinking') {
          liveThinking += ev.text;
          setTurns((t) => {
            const next = [...t];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') next[next.length - 1] = { ...last, thinking: liveThinking };
            return next;
          });
        } else if (ev.kind === 'text') {
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
          props.onTurnDone?.();
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
        {pendingAttachments.length > 0 && (
          <div className="chat-attachments">
            {pendingAttachments.map((a) => (
              <AttachmentChip
                key={a.id}
                attachment={a}
                onRemove={() => removeAttachment(a.id)}
              />
            ))}
          </div>
        )}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={pendingAttachments.length ? 'add a message (optional)…' : 'message…'}
          rows={2}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          onPaste={(e) => {
            const files: File[] = [];
            for (const item of Array.from(e.clipboardData.items)) {
              if (item.kind === 'file') {
                const f = item.getAsFile();
                if (f) files.push(f);
              }
            }
            if (files.length) {
              e.preventDefault();
              addFiles(files);
            }
          }}
          disabled={busy}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/gif,image/webp,application/pdf"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <div className="chat-actions">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={busy}
            title="attach image or PDF"
          >
            attach
          </button>
          {busy ? (
            <button onClick={cancel}>cancel</button>
          ) : (
            <button
              onClick={send}
              disabled={
                (!input.trim() && pendingAttachments.length === 0)
                || uploadsPending || uploadsFailed
              }
              title={
                uploadsPending ? 'waiting for upload…'
                : uploadsFailed ? 'remove failed attachments'
                : undefined
              }
            >
              send
            </button>
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
        {(turn.attachments?.length ?? 0) > 0 && (
          <div className="chat-attachments">
            {turn.attachments!.map((a) => (
              <AttachmentChip key={a.id} attachment={a} />
            ))}
          </div>
        )}
        {turn.text && <div className="msg-text">{turn.text}</div>}
      </div>
    );
  }
  return (
    <div className="msg msg-assistant">
      <div className="msg-role">copilot {turn.streaming && <span className="streaming">…</span>}</div>
      {turn.thinking && (
        <details className="thinking-block">
          <summary>💭 thinking</summary>
          <div className="thinking-body">{turn.thinking}</div>
        </details>
      )}
      {(turn.blocks ?? []).map((b, i) => {
        if (b.type === 'text') {
          return (
            <div key={i} className="msg-text markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{b.text}</ReactMarkdown>
            </div>
          );
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

function AttachmentChip({
  attachment, onRemove,
}: { attachment: Attachment; onRemove?: () => void }) {
  const status = attachment.uploading
    ? 'uploading…'
    : attachment.uploadError
      ? `failed: ${attachment.uploadError}`
      : humanSize(attachment.size);
  const cls = [
    'attachment',
    `attachment-${attachment.kind}`,
    attachment.uploading ? 'attachment-uploading' : '',
    attachment.uploadError ? 'attachment-error' : '',
  ].filter(Boolean).join(' ');
  return (
    <div className={cls}>
      {attachment.previewUrl ? (
        <img src={attachment.previewUrl} alt={attachment.filename} className="attachment-thumb" />
      ) : (
        <span className="attachment-icon">PDF</span>
      )}
      <span className="attachment-meta">
        <span className="attachment-name">{attachment.filename}</span>
        <span className="attachment-size">{status}</span>
      </span>
      {onRemove && (
        <button className="attachment-remove" onClick={onRemove} title="remove">×</button>
      )}
    </div>
  );
}
