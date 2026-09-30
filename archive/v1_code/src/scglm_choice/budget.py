"""Write-ahead accounting survives model-checkpoint rollback."""
from contextlib import contextmanager
import signal
import time

from .common import read, write_json, event, StudyStopped


class BudgetExceeded(StudyStopped):
    pass


class Budget:
    def __init__(self, directory, cfg, *, clock=time.time):
        self.directory, self.cfg, self.clock = directory, cfg, clock
        self.path = directory / "budget.json"
        if not self.path.exists() and ((directory / "preflight.json").exists()
                                      or any((directory / "runs").glob("*/run.json"))
                                      or (directory / "execution_contract.json").exists()):
            raise ValueError("Missing cumulative compute ledger; refusing to reset study resources")
        self.value = read(self.path) if self.path.exists() else {"categories": {}, "active": None, "events": 0}
        if self.value["active"]:
            self._close(crashed=True)

    def _entry(self, category):
        return self.value["categories"].setdefault(category, {"seconds": 0., "positions": 0, "by_kind": {}, "uncertain_seconds": 0.})

    def remaining(self, category):
        cap = 7200 if category == "shared" else 3600
        active = self.value["active"]
        live = max(0., self.clock()-active["started"]) if active else 0.
        category_live = live if active and active["category"] == category else 0.
        return max(0., min(cap - self._entry(category)["seconds"] - category_live,
                           28800 - sum(r["seconds"] for r in self.value["categories"].values()) - live))

    def _save(self):
        write_json(self.path, self.value)

    def _close(self, *, crashed=False):
        active = self.value["active"]
        if not active:
            return
        # Downtime after an unclean exit is conservatively charged. It is never
        # subtracted or silently refunded on a model/optimizer rollback.
        elapsed = min(max(0., self.clock() - active["started"]), active["reserved_seconds"])
        entry = self._entry(active["category"])
        entry["seconds"] += elapsed
        if crashed:
            entry["uncertain_seconds"] += elapsed
        event(self.directory, "gpu_budget_session_closed", category=active["category"], seconds=elapsed,
              conservative_crash_charge=crashed)
        self.value["active"] = None
        self._save()

    @contextmanager
    def session(self, category, *, timer=True):
        if self.value["active"]:
            raise ValueError("Nested GPU budget session")
        remaining = self.remaining(category)
        if remaining <= 0:
            raise BudgetExceeded("Cumulative assigned-GPU time exhausted")
        self.value["active"] = {"category": category, "started": self.clock(), "reserved_seconds": remaining}
        self._save()
        previous = signal.getsignal(signal.SIGALRM)
        def expired(*_):
            raise BudgetExceeded("Assigned-GPU wall-time ceiling reached")
        if timer:
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield self
        finally:
            if timer:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            self._close()

    def charge(self, positions, kind):
        if type(positions) is not int or positions < 0 or not self.value["active"]:
            raise ValueError("Forward work requires a live accounting session")
        category = self.value["active"]["category"]
        entry = self._entry(category)
        if category != "shared" and entry["positions"] + positions > 10_000_000:
            raise BudgetExceeded("Cumulative forward-position ceiling reached")
        elapsed = self.clock() - self.value["active"]["started"]
        if elapsed >= self.value["active"]["reserved_seconds"]:
            raise BudgetExceeded("Cumulative time ceiling reached")
        entry["positions"] += positions
        entry["by_kind"][kind] = entry["by_kind"].get(kind, 0) + positions
        self.value["events"] += 1
        # Charged before execution; partial/discarded forwards still cost budget.
        self._save()


def evaluation_positions(primary, raw, instructions, transfer, tokenizer, templates):
    from .objective import choice_rows, positions
    from scglm_pipeline.task_registry import prompt_text
    total = sum(positions(choice_rows(tokenizer, r, t)) for r in primary for t in templates)
    total += sum(len(r["tokens"]) for r in raw)
    total += sum(positions(choice_rows(tokenizer, r)) for r in transfer)
    # Greedy KV-cache decoding: one unpadded prompt and <=127 one-token forwards.
    total += sum(len(tokenizer.encode(prompt_text(r), add_special_tokens=False)) + 127 for r in instructions)
    return total
