"""Session store: append-only event log + git-like state DAG + blob store.

Single-user, file-backed under config.session_dir():

    <session_dir>/
      index.json                     # lightweight list for the picker
      <session_id>/
        events.jsonl                 # append-only full log (source of truth)
        tree.json                    # state DAG + HEAD + name
        blobs/<sha256>.json.gz       # content-addressed geometry/sim results

See docs/SESSIONS_DESIGN.md. This module is the storage layer only; the HTTP
routes and the logging integration into the chat/exec endpoints live elsewhere.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import session_dir

_lock = threading.RLock()
_counter = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen(prefix: str) -> str:
    """Monotonic-ish sortable id: prefix_<ms><counter><rand>."""
    global _counter
    with _lock:
        _counter = (_counter + 1) % 1000
        c = _counter
    return f"{prefix}_{int(__import__('time').time() * 1000):013d}{c:03d}{os.urandom(2).hex()}"


def _root() -> Path:
    d = session_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sdir(sid: str) -> Path:
    return _root() / sid


def _read_json(p: Path, default: Any) -> Any:
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json_atomic(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# --------------------------------------------------------------------------- #
# index (for the session picker)
# --------------------------------------------------------------------------- #
def _index_path() -> Path:
    return _root() / 'index.json'


def _index_upsert(tree: dict) -> None:
    with _lock:
        idx = _read_json(_index_path(), {'sessions': []})
        entry = {
            'id': tree['session_id'], 'name': tree['name'],
            'name_source': tree.get('name_source', 'auto'),
            'created': tree['created'], 'updated': tree['updated'],
            'n_nodes': len(tree['nodes']),
        }
        rest = [s for s in idx['sessions'] if s['id'] != tree['session_id']]
        idx['sessions'] = [entry] + rest
        _write_json_atomic(_index_path(), idx)


def list_sessions() -> list[dict]:
    idx = _read_json(_index_path(), {'sessions': []})
    return sorted(idx['sessions'], key=lambda s: s.get('updated', ''), reverse=True)


# --------------------------------------------------------------------------- #
# session lifecycle
# --------------------------------------------------------------------------- #
def create_session(name: Optional[str] = None, model: Optional[str] = None) -> dict:
    sid = _gen('sess')
    d = _sdir(sid)
    (d / 'blobs').mkdir(parents=True, exist_ok=True)
    open(d / 'events.jsonl', 'a').close()
    now = _now()
    root = _gen('node')
    tree = {
        'session_id': sid,
        'name': name or 'Untitled session',
        'name_source': 'user' if name else 'auto',
        'created': now, 'updated': now,
        'head': root, 'model': model,
        'nodes': {
            root: {
                'id': root, 'parent': None, 'children': [], 'ts': now,
                'kind': 'root', 'label': 'session start', 'event_ids': [],
                'snapshot': {'code': None, 'code_hash': None,
                             'geometry_ref': None, 'sim_ref': None, 'chat_len': 0},
            }
        },
    }
    _write_json_atomic(d / 'tree.json', tree)
    append_event(sid, 'session_meta',
                 {'name': tree['name'], 'name_source': tree['name_source'],
                  'model': model, 'created': now}, node_id=root)
    _index_upsert(tree)
    return tree


def get_tree(sid: str) -> Optional[dict]:
    return _read_json(_sdir(sid) / 'tree.json', None)


def delete_session(sid: str) -> bool:
    d = _sdir(sid)
    if not d.is_dir():
        return False
    shutil.rmtree(d)
    with _lock:
        idx = _read_json(_index_path(), {'sessions': []})
        idx['sessions'] = [s for s in idx['sessions'] if s['id'] != sid]
        _write_json_atomic(_index_path(), idx)
    return True


def set_name(sid: str, name: str, source: str) -> Optional[dict]:
    """source 'user' always wins; 'auto' never overwrites a user-set name."""
    with _lock:
        tree = get_tree(sid)
        if tree is None:
            return None
        if source == 'auto' and tree.get('name_source') == 'user':
            return tree
        tree['name'] = name
        tree['name_source'] = source
        tree['updated'] = _now()
        _write_json_atomic(_sdir(sid) / 'tree.json', tree)
        _index_upsert(tree)
    return tree


# --------------------------------------------------------------------------- #
# events (append-only)
# --------------------------------------------------------------------------- #
def append_event(sid: str, type: str, payload: dict,
                 node_id: Optional[str] = None,
                 parent_node: Optional[str] = None) -> dict:
    ev = {'id': _gen('evt'), 'ts': _now(), 'node_id': node_id,
          'parent_node': parent_node, 'type': type, 'payload': payload}
    with _lock:
        with open(_sdir(sid) / 'events.jsonl', 'a') as f:
            f.write(json.dumps(ev) + '\n')
            f.flush()
            os.fsync(f.fileno())
    return ev


def read_events(sid: str, node_id: Optional[str] = None,
                types: Optional[set] = None) -> Iterator[dict]:
    p = _sdir(sid) / 'events.jsonl'
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if node_id is not None and ev.get('node_id') != node_id:
                continue
            if types is not None and ev.get('type') not in types:
                continue
            yield ev


# --------------------------------------------------------------------------- #
# state DAG nodes (git-like checkpoints)
# --------------------------------------------------------------------------- #
def add_node(sid: str, kind: str, label: str, snapshot: dict,
             event_ids: Optional[list] = None,
             parent: Optional[str] = None) -> Optional[dict]:
    """Create a child checkpoint of `parent` (default = current HEAD) and
    advance HEAD to it. Acting after a checkout to a non-leaf forks a branch."""
    with _lock:
        tree = get_tree(sid)
        if tree is None:
            return None
        parent = parent or tree['head']
        nid = _gen('node')
        node = {'id': nid, 'parent': parent, 'children': [], 'ts': _now(),
                'kind': kind, 'label': label, 'event_ids': event_ids or [],
                'snapshot': snapshot}
        tree['nodes'][nid] = node
        if parent in tree['nodes']:
            tree['nodes'][parent]['children'].append(nid)
        tree['head'] = nid
        tree['updated'] = _now()
        _write_json_atomic(_sdir(sid) / 'tree.json', tree)
        _index_upsert(tree)
    return node


def checkout(sid: str, node_id: str) -> Optional[dict]:
    """Move HEAD to an existing node (rewind). Returns that node."""
    with _lock:
        tree = get_tree(sid)
        if tree is None or node_id not in tree['nodes']:
            return None
        tree['head'] = node_id
        tree['updated'] = _now()
        _write_json_atomic(_sdir(sid) / 'tree.json', tree)
        _index_upsert(tree)
        return tree['nodes'][node_id]


def path_to_root(sid: str, node_id: str) -> list[str]:
    """node ids from root → node_id (the conversation prefix for that node)."""
    tree = get_tree(sid)
    if tree is None or node_id not in tree['nodes']:
        return []
    chain = []
    cur = node_id
    while cur is not None:
        chain.append(cur)
        cur = tree['nodes'].get(cur, {}).get('parent')
    return list(reversed(chain))


# --------------------------------------------------------------------------- #
# chat transcript reconstruction (for rehydrating the chat UI on restore)
# --------------------------------------------------------------------------- #
def _parse_user_content(content) -> tuple[str, list]:
    """A logged user_message content (str | list of wire blocks) → (text,
    attachments) shaped for the frontend ChatTurn."""
    if isinstance(content, str):
        return content, []
    text_parts: list[str] = []
    atts: list[dict] = []
    for b in (content or []):
        if not isinstance(b, dict):
            continue
        bt = b.get('type')
        if bt == 'text':
            text_parts.append(b.get('text', ''))
        elif bt == 'image':
            src = b.get('source', {}) or {}
            if src.get('type') == 'base64':
                mt = src.get('media_type', 'image/png')
                data = src.get('data', '')
                atts.append({'id': _gen('att'), 'kind': 'image', 'mediaType': mt,
                             'filename': 'image', 'size': 0, 'dataB64': data,
                             'previewUrl': f'data:{mt};base64,{data}'})
        elif bt == 'document':
            src = b.get('source', {}) or {}
            att = {'id': _gen('att'), 'kind': 'document',
                   'mediaType': src.get('media_type', 'application/pdf'),
                   'filename': b.get('title') or b.get('name') or 'document.pdf',
                   'size': 0}
            if src.get('type') == 'file':
                att['fileId'] = src.get('file_id')
            elif src.get('type') == 'base64':
                att['dataB64'] = src.get('data', '')
            atts.append(att)
    return ' '.join(t for t in text_parts if t).strip(), atts


def transcript(sid: str, node_id: Optional[str] = None) -> list[dict]:
    """Rebuild the chat transcript (frontend ChatTurn[] shape) from the event
    log, for the conversation prefix ending at `node_id` (default HEAD). Lets
    the UI rehydrate chat history + attachments after a reload/checkout — the
    backend is the source of truth; the live frontend only mirrors it."""
    tree = get_tree(sid)
    if tree is None:
        return []
    target = node_id or tree.get('head')
    allowed: Optional[set] = None
    if target and target in (tree.get('nodes') or {}):
        ids: list = []
        for nid in path_to_root(sid, target):
            ids.extend(tree['nodes'].get(nid, {}).get('event_ids', []))
        allowed = set(ids)

    decisions: dict[str, str] = {}    # proposal summary → applied|discarded
    turns: list[dict] = []
    cur: Optional[dict] = None
    for ev in read_events(sid):
        if allowed is not None and ev.get('id') not in allowed:
            continue
        t = ev.get('type')
        p = ev.get('payload') or {}
        if t == 'proposal_decision':
            decisions[p.get('summary', '')] = (
                'applied' if p.get('status') in ('accepted', 'applied') else 'discarded')
        elif t == 'user_message':
            text, atts = _parse_user_content(p.get('content'))
            turns.append({'id': ev['id'], 'role': 'user', 'text': text,
                          'attachments': atts})
            cur = {'id': _gen('turn'), 'role': 'assistant', 'blocks': [],
                   'thinking': '', 'proposals': [], 'toolResults': [],
                   'streaming': False}
            turns.append(cur)
        elif t == 'copilot_response' and cur is not None:
            blocks: list[dict] = []
            for b in p.get('content_blocks', []):
                bt = b.get('type')
                if bt == 'thinking':
                    cur['thinking'] += b.get('thinking', '') or ''
                elif bt == 'text':
                    blocks.append({'type': 'text', 'text': b.get('text', '')})
                elif bt == 'tool_use':
                    blocks.append({'type': 'tool_use', 'id': b.get('id'),
                                   'name': b.get('name'), 'input': b.get('input', {})})
            if blocks:    # last response's display blocks win (matches live UI)
                cur['blocks'] = blocks
        elif t == 'proposal' and cur is not None:
            cur['proposals'].append({'id': p.get('tool_id') or _gen('prop'),
                                     'new_code': p.get('new_code', ''),
                                     'summary': p.get('summary', ''),
                                     'status': 'pending'})
        elif t == 'tool_exec' and cur is not None:
            cur['toolResults'].append({'tool_id': p.get('tool_id'),
                                       'name': p.get('name'),
                                       'result': p.get('result')})

    for tn in turns:
        for pr in tn.get('proposals', []):
            if pr['summary'] in decisions:
                pr['status'] = decisions[pr['summary']]
    # drop a trailing assistant turn that captured nothing (interrupted/errored)
    while turns and turns[-1]['role'] == 'assistant' and not turns[-1]['blocks'] \
            and not turns[-1]['proposals'] and not turns[-1]['toolResults'] \
            and not turns[-1]['thinking']:
        turns.pop()
    return turns


# --------------------------------------------------------------------------- #
# content-addressed blobs (dedup'd geometry/sim results incl. meshes)
# --------------------------------------------------------------------------- #
def put_blob(sid: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    h = hashlib.sha256(raw).hexdigest()
    p = _sdir(sid) / 'blobs' / f'{h}.json.gz'
    if not p.exists():
        with gzip.open(p, 'wb') as f:
            f.write(raw)
    return f'blob_{h}'


def get_blob(sid: str, ref: str) -> Optional[dict]:
    h = ref[len('blob_'):] if ref.startswith('blob_') else ref
    p = _sdir(sid) / 'blobs' / f'{h}.json.gz'
    if not p.exists():
        return None
    with gzip.open(p, 'rb') as f:
        return json.loads(f.read())


def session_size(sid: str) -> int:
    total = 0
    for dp, _, files in os.walk(_sdir(sid)):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- #
# cleanup / retention (P5)
# --------------------------------------------------------------------------- #
def usage() -> dict:
    """Per-session disk usage + total, for the cleanup view."""
    out = []
    total = 0
    for s in list_sessions():
        sz = session_size(s['id'])
        total += sz
        out.append({**s, 'size_bytes': sz})
    return {'sessions': out, 'total_bytes': total}


def _blob_refs_in(obj) -> set:
    """Collect any 'blob_...' references nested anywhere in a JSON-like value."""
    refs = set()
    def walk(v):
        if isinstance(v, str) and v.startswith('blob_'):
            refs.add(v[len('blob_'):])
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(obj)
    return refs


def prune_to_branch(sid: str) -> Optional[dict]:
    """Keep only the root→HEAD lineage (the current branch); drop sibling
    branches, their events, and orphaned blobs. Returns the new tree."""
    with _lock:
        tree = get_tree(sid)
        if tree is None:
            return None
        keep = set(path_to_root(sid, tree['head']))
        if not keep:
            return tree
        # rewrite nodes: keep lineage; fix children to only kept ids
        new_nodes = {}
        for nid in keep:
            n = dict(tree['nodes'][nid])
            n['children'] = [c for c in n['children'] if c in keep]
            new_nodes[nid] = n
        tree['nodes'] = new_nodes
        tree['updated'] = _now()
        _write_json_atomic(_sdir(sid) / 'tree.json', tree)
        # rewrite events.jsonl keeping events for kept nodes (or node-less)
        p = _sdir(sid) / 'events.jsonl'
        kept_refs = set()
        for nid in keep:
            snap = new_nodes[nid].get('snapshot', {})
            for r in (snap.get('geometry_ref'), snap.get('sim_ref')):
                if r:
                    kept_refs.add(r[len('blob_'):] if r.startswith('blob_') else r)
        if p.exists():
            kept_lines = []
            for line in open(p):
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get('node_id') in keep or ev.get('node_id') is None:
                    kept_lines.append(line.rstrip('\n'))
                    kept_refs |= _blob_refs_in(ev.get('payload'))
            tmp = p.with_suffix('.jsonl.tmp')
            with open(tmp, 'w') as f:
                f.write('\n'.join(kept_lines) + ('\n' if kept_lines else ''))
            os.replace(tmp, p)
        # GC blobs not referenced by kept nodes/events
        bdir = _sdir(sid) / 'blobs'
        if bdir.is_dir():
            for f in bdir.glob('*.json.gz'):
                if f.name[:-len('.json.gz')] not in kept_refs:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        _index_upsert(tree)
        return tree


def delete_older(sid: str) -> int:
    """Delete this session and every session last updated at or before it.
    Returns the number of sessions removed."""
    idx = list_sessions()
    target = next((s for s in idx if s['id'] == sid), None)
    if target is None:
        return 0
    cutoff = target['updated']
    victims = [s['id'] for s in idx if s.get('updated', '') <= cutoff]
    n = 0
    for vid in victims:
        if delete_session(vid):
            n += 1
    return n
