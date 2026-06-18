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
