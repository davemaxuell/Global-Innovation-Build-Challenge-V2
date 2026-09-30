from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jsonl(path):
    """Ignore only an unfinished last log line written by the live trainer."""
    raw = Path(path).read_bytes()
    complete = raw[:raw.rfind(b'\n') + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Artifact checksum changed: {path}')
    return read(path)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def artifact_hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify_receipt(receipt, registration):
    if receipt['registration_sha256'] != registration:
        raise ValueError('Stage receipt belongs to another registration')
    folder = Path(receipt['output'])
    if artifact_hashes(folder) != receipt['artifacts']:
        raise ValueError(f'Completed stage artifacts changed: {folder}')
    if receipt.get('status') != 'completed':
        raise ValueError('Incomplete stage receipt')


@contextmanager
def lease(path, *, inherited_fd=None):
    """Nonblocking flock; an inherited descriptor shares the parent's lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if inherited_fd is None:
        f = path.open('a')
    else:
        f = os.fdopen(os.dup(inherited_fd), 'a')
        stat = path.stat()
        actual = os.fstat(f.fileno())
        if (stat.st_ino, stat.st_dev) != (actual.st_ino, actual.st_dev):
            f.close()
            raise ValueError('Inherited GPU lease is for a different file')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield f
    finally:
        # Do not LOCK_UN an inherited open-file description: parent owns it.
        f.close()
