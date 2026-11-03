#!/usr/bin/env python3
"""Generate four charts from a CSV produced by src/main.py."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/plot_results.py results/benchmark_....csv")
    source = Path(sys.argv[1])
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV contains no result rows")
    concurrency = [int(r["concurrency"]) for r in rows]
    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)
    charts = [
        ("throughput", "throughput", "System output throughput (tokens/s)"),
        ("latency", "avg_latency", "Mean request latency (s)"),
        ("ttft", "avg_ttft_ms", "Mean client-observed TTFT (ms)"),
        ("tpot", "avg_tpot_ms", "Mean client-estimated TPOT (ms)"),
    ]
    for suffix, key, ylabel in charts:
        x, y = [], []
        for c, r in zip(concurrency, rows):
            value = r.get(key)
            if value not in (None, "", "None"):
                x.append(c)
                y.append(float(value))
        if not x:
            print(f"跳过 {suffix}: no numeric data")
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Concurrent requests")
        plt.ylabel(ylabel)
        plt.title(f"vLLM {suffix.upper()} vs concurrency")
        plt.grid(True)
        plt.tight_layout()
        path = output_dir / f"{source.stem}_{suffix}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"已保存：{path}")


if __name__ == "__main__":
    main()
