"""Deterministic sequential token streams with explicit resumable cursors."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import numpy as np


class SourceStream:
    """Read uint16 shards as one circular sequence; every output has ``seq_len`` targets.

    Adjacent examples overlap by one input token, so targets are neither skipped nor
    double counted. A cycle is a wrap of the underlying source token stream.
    """

    def __init__(self, paths: str | Path | Sequence[str | Path], seq_len: int = 1024):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [str(Path(p).resolve()) for p in paths]
        if not self.paths or seq_len < 1:
            raise ValueError("At least one shard and a positive seq_len are required")
        for path in self.paths:
            if Path(path).stat().st_size % 2:
                raise ValueError(f"Invalid uint16 shard byte length: {path}")
        self.arrays = [np.memmap(p, dtype="<u2", mode="r") for p in self.paths]
        self.lengths = [len(a) for a in self.arrays]
        self.ends = np.cumsum(self.lengths)
        self.total_tokens = int(sum(self.lengths))
        if self.total_tokens < 2:
            raise ValueError("Source must contain at least two tokens")
        self.seq_len = int(seq_len)
        self.cursor = 0
        self.cycles = 0
        self.tokens_consumed = 0
        signature = repr(list(zip(self.paths, self.lengths))).encode()
        self.signature = hashlib.sha256(signature).hexdigest()

    def _read(self, position: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.int64)
        written = 0
        while written < length:
            pos = position % self.total_tokens
            shard = int(np.searchsorted(self.ends, pos, side="right"))
            start = 0 if shard == 0 else int(self.ends[shard - 1])
            local = pos - start
            count = min(length - written, self.lengths[shard] - local)
            out[written:written + count] = self.arrays[shard][local:local + count]
            position += count
            written += count
        return out

    def next_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Batch size must be positive")
        count = int(n) * self.seq_len
        data = self._read(self.cursor, count + 1)
        x = data[:-1].reshape(n, self.seq_len).copy()
        y = data[1:].reshape(n, self.seq_len).copy()
        advanced = self.cursor + count
        self.cycles += advanced // self.total_tokens
        self.cursor = advanced % self.total_tokens
        self.tokens_consumed += count
        return x, y

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "cycles": self.cycles,
                "tokens_consumed": self.tokens_consumed, "seq_len": self.seq_len,
                "total_tokens": self.total_tokens, "signature": self.signature}

    def load_state_dict(self, state: dict) -> None:
        for key in ("seq_len", "total_tokens", "signature"):
            if state[key] != getattr(self, key):
                raise ValueError(f"SourceStream state mismatch: {key}")
        cursor, cycles, consumed = (int(state[k]) for k in ("cursor", "cycles", "tokens_consumed"))
        if not 0 <= cursor < self.total_tokens or cycles < 0 or consumed < 0:
            raise ValueError("Invalid stream cursor/cycle state")
        if consumed != cycles * self.total_tokens + cursor:
            raise ValueError("Inconsistent stream token accounting")
        self.cursor, self.cycles, self.tokens_consumed = cursor, cycles, consumed


def iter_documents(path: str | Path):
    """Read prepared evaluation JSONL without loading all documents into memory."""
    import json
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
