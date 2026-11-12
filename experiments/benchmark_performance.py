"""
Benchmark de rendimiento con archivos grandes (hasta ~1–5M filas según RAM).

Ejecutar: python experiments/benchmark_performance.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estacion.core.profiler import profile_dataframe
from estacion.discovery.engine import run_discovery
from estacion.stats.correlation import correlation_analysis
from estacion.clustering.segmenter import run_segmentation


def make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "region": rng.choice(["N", "S", "E", "W"], n),
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "z": rng.lognormal(0, 1, n),
            "flag": rng.integers(0, 2, n),
            "date": pd.date_range("2018-01-01", periods=n, freq="min"),
        }
    )


def timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.2f}s")
    return out, dt


def main():
    sizes = [50_000, 200_000, 1_000_000]
    # 5M solo si hay flag
    if "--large" in sys.argv:
        sizes.append(5_000_000)

    results = []
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    for n in sizes:
        print(f"\n=== n={n:,} ===")
        df, t_gen = timed("generate", lambda: make_frame(n))
        csv_path = out_dir / f"bench_{n}.csv"
        _, t_csv = timed("to_csv", lambda: df.to_csv(csv_path, index=False))
        df2, t_read = timed("read_csv", lambda: pd.read_csv(csv_path, parse_dates=["date"]))
        _, t_prof = timed("profile", lambda: profile_dataframe(df2, f"bench_{n}"))
        # discovery sobre muestra interna ya acotada; medir igual
        sample = df2 if n <= 200_000 else df2.sample(200_000, random_state=0)
        findings, _, art, = None, None, None  # noqa

        def _disc():
            return run_discovery(sample, f"bench_{n}")

        (findings, profile, art), t_disc = timed("discovery_on_sample", _disc)
        nums = ["x", "y", "z"]
        _, t_corr = timed("correlation", lambda: correlation_analysis(sample, nums))
        _, t_seg = timed(
            "segmentation",
            lambda: run_segmentation(sample.sample(min(8000, len(sample)), random_state=1), nums),
        )

        row = {
            "n": n,
            "t_generate_s": t_gen,
            "t_to_csv_s": t_csv,
            "t_read_csv_s": t_read,
            "t_profile_s": t_prof,
            "discovery_sample_n": len(sample),
            "t_discovery_s": t_disc,
            "n_findings": len(findings),
            "t_correlation_s": t_corr,
            "t_segmentation_s": t_seg,
            "csv_mb": csv_path.stat().st_size / (1024 * 1024),
        }
        results.append(row)
        # limpiar CSV enorme
        if n >= 1_000_000:
            csv_path.unlink(missing_ok=True)

    summary = {"results": results, "note": "discovery usa muestra ≤200k; clustering ≤8k"}
    path = out_dir / "benchmark_report.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nWrote", path)
    print(pd.DataFrame(results).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
