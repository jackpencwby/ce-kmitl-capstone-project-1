"""GBR iteration progress; percentages count trees, not elapsed wall time."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def write_atomic(path, payload, *, best_effort=False):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Windows may briefly deny replacement while another worker reads the file.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                if best_effort:
                    temporary.unlink(missing_ok=True)
                    return False
                raise
            time.sleep(min(0.01 * 2 ** attempt, 0.25))
    return True


def summarize(checkpoint, horizons, total_trees):
    rows = []
    for h in horizons:
        if (checkpoint / f"h{h}.joblib").exists():
            row = {"horizon": h, "state": "completed", "trees_done": total_trees}
        else:
            try:
                row = json.loads((checkpoint / f"h{h}_progress.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                row = {"horizon": h, "state": "unknown_or_pending", "trees_done": 0}
        rows.append(row)
    done = sum(min(total_trees, max(0, r["trees_done"])) for r in rows)
    return {"training_percent": 100 * done / (len(horizons) * total_trees),
            "trees_done": done, "trees_total": len(horizons) * total_trees,
            "horizons": rows}


class GBRProgress:
    def __init__(self, checkpoint, horizon, horizons, total_trees, log_every=10):
        self.checkpoint = Path(checkpoint)
        self.horizon = horizon
        self.horizons = list(horizons)
        self.total = total_trees
        self.log_every = log_every
        self.start = time.monotonic()
        self.row = {"horizon": horizon, "pid": os.getpid(), "trees_done": 0,
                    "trees_total": total_trees, "state": "preparing"}
        self.publish()

    def publish(self):
        self.row["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.row["training_percent"] = 100 * self.row["trees_done"] / self.total
        # Progress is observability only; a transient Windows/antivirus file lock
        # must never abort an hours-long model fit.
        write_atomic(self.checkpoint / f"h{self.horizon}_progress.json", self.row,
                     best_effort=True)

    def __call__(self, iteration, estimator, local_variables):
        done = iteration + 1
        elapsed = time.monotonic() - self.start
        self.row.update(trees_done=done, state="training" if done < self.total else "predicting",
                        elapsed_seconds=round(elapsed, 1),
                        estimated_training_seconds_remaining=round(elapsed / done * (self.total - done), 1))
        # Small atomic JSON after every tree; human-readable log every N trees.
        self.publish()
        if done == 1 or done % self.log_every == 0 or done == self.total:
            overall = summarize(self.checkpoint, self.horizons, self.total)
            print(f"[{self.row['updated_at']}] h{self.horizon}: {done}/{self.total} trees "
                  f"({100 * done / self.total:.1f}%) | overall training "
                  f"{overall['training_percent']:.1f}% | elapsed {elapsed / 60:.1f} min | "
                  f"horizon training ETA ~{self.row['estimated_training_seconds_remaining'] / 60:.1f} min",
                  flush=True)
        return False

    def finish(self, state, error=None):
        self.row["state"] = state
        if error is not None:
            self.row["error"] = str(error)
        self.publish()
