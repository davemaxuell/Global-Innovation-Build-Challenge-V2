"""Read-only helpers for the portable competition package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inventory(root):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Package must contain independent files, not symlinks: {p}')
        if p.is_file() and p.relative_to(root).as_posix() != 'MANIFEST.json':
            files[p.relative_to(root).as_posix()] = sha256(p)
    return files


def verify(root):
    root = Path(root).resolve()
    manifest = read(root/'MANIFEST.json')
    if manifest['files'] != inventory(root):
        raise ValueError('Package inventory/checksums differ from MANIFEST.json')
    weight_hash = sha256(root/'model/model.safetensors')
    if weight_hash != manifest['identity']['model_sha256']:
        raise ValueError('Packaged weights differ from selected endpoint')
    selection = read(root/'results/selection.json')
    if selection['selected_model_sha256'] != weight_hash or selection['official_scores_used'] is not False:
        raise ValueError('Selection identity or policy differs')
    return manifest


def outside_package(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root in path.parents:
        raise ValueError('Write evaluation/demo output outside the immutable package')
    return path
