"""Metric logging to CSV and TensorBoard."""

from __future__ import annotations

import csv
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard is optional on shared HPC environments.
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


class MetricLogger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(str(self.out_dir / "tb"))
        self._csv_path = self.out_dir / "metrics.csv"
        self._fields: list[str] | None = None

    def log(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        row = {f"{prefix}{k}": v for k, v in metrics.items()}
        for k, v in row.items():
            self.tb.add_scalar(k, v, step)
        row = {"step": step, **row}
        new_fields = sorted(set(row) | set(self._fields or []))
        rewrite = self._fields is not None and new_fields != self._fields
        if rewrite:
            rows = list(csv.DictReader(self._csv_path.open()))
        if self._fields is None or rewrite:
            self._fields = new_fields
            with self._csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fields, restval="")
                w.writeheader()
                if rewrite:
                    w.writerows(rows)
        with self._csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields, restval="").writerow(row)

    def close(self) -> None:
        self.tb.close()
