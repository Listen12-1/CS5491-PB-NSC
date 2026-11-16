"""Measure development scoring time in fresh Python processes."""

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPEATS = 7

METHODS = [
    "original_nsc",
    "attention_only",
    "ffn_only",
    "matrix_wise",
    "pb_nsc",
]


def worker(method):
    # Imports and data loading are outside the scoring timer.
    from nsc import original_nsc
    from pb_nsc_score import (
        attention_only_score,
        ffn_only_score,
        matrix_wise_score,
        score,
    )

    scorers = {
        "original_nsc": original_nsc,
        "attention_only": attention_only_score,
        "ffn_only": ffn_only_score,
        "matrix_wise": matrix_wise_score,
        "pb_nsc": score,
    }

    configs = json.loads(
        (ROOT / "data" / "architectures.json").read_text(
            encoding="utf-8"
        )
    )
    splits = json.loads(
        (ROOT / "data" / "splits.json").read_text(
            encoding="utf-8"
        )
    )

    selected_configs = [
        configs[key] for key in splits["development"]
    ]

    scorer = scorers[method]

    # Fresh process: no previous method has populated the MP cache.
    # Cache reuse within this 100-architecture batch is allowed.
    started = time.perf_counter()
    values = [scorer(config) for config in selected_configs]
    scoring_seconds = time.perf_counter() - started

    if not all(math.isfinite(value) for value in values):
        raise ValueError("A scorer returned a non-finite value.")

    print(json.dumps({
        "method": method,
        "architectures": len(values),
        "scoring_seconds": scoring_seconds,
        "score_sum": math.fsum(values),
    }))


def benchmark():
    records = {name: [] for name in METHODS}
    failures = []

    for repetition in range(REPEATS):
        # Rotate execution order to reduce a fixed order effect.
        offset = repetition % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]

        for method in order:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                method,
            ]

            started = time.perf_counter()
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            process_seconds = time.perf_counter() - started

            if process.returncode != 0:
                failures.append({
                    "method": method,
                    "repetition": repetition + 1,
                    "stderr": process.stderr,
                })
                print(f"FAILED: {method}, run {repetition + 1}")
                continue

            record = json.loads(process.stdout)
            record["repetition"] = repetition + 1
            record["process_seconds"] = process_seconds
            records[method].append(record)

        print(
            f"Completed round {repetition + 1}/{REPEATS}",
            flush=True,
        )

    summaries = {}

    print("\nFresh-process scoring of 100 development architectures:")

    for method in METHODS:
        runs = records[method]
        times = [run["scoring_seconds"] for run in runs]

        if not times:
            summaries[method] = {"successful_runs": 0}
            continue

        median_time = statistics.median(times)

        summaries[method] = {
            "successful_runs": len(runs),
            "median_scoring_seconds": median_time,
            "min_scoring_seconds": min(times),
            "max_scoring_seconds": max(times),
            "median_ms_per_architecture": median_time * 1000 / 100,
        }

        print(
            f"{method:16s} "
            f"median={median_time:.4f}s "
            f"range=[{min(times):.4f}, {max(times):.4f}]s"
        )

    baseline = summaries["original_nsc"].get(
        "median_scoring_seconds"
    )

    if baseline:
        for summary in summaries.values():
            if "median_scoring_seconds" in summary:
                summary["ratio_to_nsc"] = (
                    summary["median_scoring_seconds"] / baseline
                )

    output = {
        "split": "development",
        "architectures": 100,
        "repeats_per_method": REPEATS,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor_reported_by_python": platform.processor(),
        "timing_protocol": (
            "Each method and repeat runs in a fresh Python process. "
            "Scoring timer excludes imports, data loading, validation "
            "and process startup. MP cache reuse within each "
            "100-architecture batch is allowed."
        ),
        "summaries": summaries,
        "failures": failures,
        "raw_runs": records,
    }

    output_path = ROOT / "benchmark_development_results.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nFailures: {len(failures)}")
    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=METHODS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker)
    else:
        benchmark()