"""Write-ahead GPU-time and forward-position accounting with configurable ceilings.

Generalizes the choice study's ledger: each category has its own time cap, an
overall cap bounds the whole study, and charges are persisted before the work
they pay for. Downtime after an unclean exit is charged, never refunded.
"""
from contextlib import contextmanager
import signal
import time

from .common import StudyStopped, event, read, write_json


class BudgetExceeded(StudyStopped):
    pass


class Budget:
    def __init__(self, directory, limits, *, clock=time.time):
        """``limits``: {"gpu_seconds": {category|"default": s}, "overall_gpu_seconds": s,
        optional "max_positions": {category|"default": n}}."""
        self.directory, self.limits, self.clock = directory, limits, clock
        self.path = directory / "budget.json"
        directory.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() and ((directory / "preflight.json").exists()
                                      or (directory / "execution_contract.json").exists()):
            raise ValueError("Missing cumulative compute ledger; refusing to reset study resources")
        self.value = read(self.path) if self.path.exists() else {"categories": {}, "active": None, "events": 0}
        if self.value["active"]:
            self._close(crashed=True)

    def _cap(self, key, category):
        table = self.limits.get(key, {})
        return table.get(category, table.get("default"))

    def _entry(self, category):
        return self.value["categories"].setdefault(
            category, {"seconds": 0., "positions": 0, "by_kind": {}, "uncertain_seconds": 0.})

    def remaining(self, category):
        cap = self._cap("gpu_seconds", category)
        if cap is None:
            raise ValueError(f"No registered GPU-time ceiling for {category}")
        active = self.value["active"]
        live = max(0., self.clock() - active["started"]) if active else 0.
        category_live = live if active and active["category"] == category else 0.
        spent = sum(r["seconds"] for r in self.value["categories"].values())
        return max(0., min(cap - self._entry(category)["seconds"] - category_live,
                           self.limits["overall_gpu_seconds"] - spent - live))

    def _save(self):
        write_json(self.path, self.value)

    def _close(self, *, crashed=False):
        active = self.value["active"]
        if not active:
            return
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
            raise BudgetExceeded(f"Cumulative assigned-GPU time exhausted: {category}")
        self.value["active"] = {"category": category, "started": self.clock(), "reserved_seconds": remaining}
        self._save()
        previous = signal.getsignal(signal.SIGALRM)

        def expired(*_):
            raise BudgetExceeded(f"Assigned-GPU wall-time ceiling reached: {category}")
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
        cap = self._cap("max_positions", category)
        if cap is not None and entry["positions"] + positions > cap:
            raise BudgetExceeded(f"Cumulative forward-position ceiling reached: {category}")
        if self.clock() - self.value["active"]["started"] >= self.value["active"]["reserved_seconds"]:
            raise BudgetExceeded(f"Cumulative time ceiling reached: {category}")
        entry["positions"] += positions
        entry["by_kind"][kind] = entry["by_kind"].get(kind, 0) + positions
        self.value["events"] += 1
        self._save()
