#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate summary CSVs and plot scheduler comparison.")
    p.add_argument("--summaries-dir", default="results/summaries", help="Directory with summary CSVs")
    p.add_argument("--metric", default="n_success", help="Metric column to aggregate (single)")
    p.add_argument("--metrics", default=None, help="Comma-separated metric columns (multi)")
    p.add_argument("--out", default="scheduler_compare.png", help="Output filename (single)")
    p.add_argument("--out-dir", default="results/summaries", help="Base output dir (timestamped subdir will be created)")
    p.add_argument("--min-seed", type=int, default=None, help="Filter: minimum seed (inclusive)")
    p.add_argument("--max-seed", type=int, default=None, help="Filter: maximum seed (inclusive)")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    summ_dir = Path(args.summaries_dir)
    if not summ_dir.exists():
        raise SystemExit(f"Summaries dir not found: {summ_dir}")

    rows: List[Dict[str, str]] = []
    for p in sorted(summ_dir.glob("summary_*.csv")):
        rows.extend(load_rows(p))

    if not rows:
        raise SystemExit("No summary CSVs found.")

    metrics: List[str]
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = [args.metric]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(args.out_dir) / ts
    base_out_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        if metric not in rows[0]:
            raise SystemExit(f"Unknown metric: {metric}")

        data: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            seed = int(float(r.get("seed", "0")))
            if args.min_seed is not None and seed < args.min_seed:
                continue
            if args.max_seed is not None and seed > args.max_seed:
                continue
            sched = r.get("scheduler", "unknown")
            data[sched].append(float(r[metric]))

        if not data:
            raise SystemExit("No rows matched seed filter.")

        # Compute means
        scheds = sorted(data.keys())
        means = [sum(vals) / len(vals) for vals in (data[s] for s in scheds)]

        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise SystemExit("matplotlib is required for plotting.") from e

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(scheds, means, color="#4C78A8")
        ax.set_ylabel(f"mean({metric})")
        ax.set_title(f"Scheduler comparison: {metric}")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()

        if args.metrics:
            out_path = base_out_dir / f"scheduler_compare_{metric}.png"
        else:
            out = Path(args.out)
            out_path = base_out_dir / out.name
        fig.savefig(out_path)
        print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
