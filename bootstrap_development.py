"""Paired, width-stratified architecture bootstrap on development only."""

import json
import random
import time
from pathlib import Path

from evaluate import canonical, correlation, pair_summary, ranks
from nsc import original_nsc
from pb_nsc_score import matrix_wise_score, score


ROOT = Path(__file__).resolve().parent
REPEATS = 2000
SEED = 5491

METRICS = ["spearman_rho", "tau_b", "controlled_tau_b"]


def load_data(filename):
    return json.loads(
        (ROOT / "data" / filename).read_text(encoding="utf-8")
    )


def percentile(values, probability):
    """Linearly interpolated sample percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def main():
    started = time.perf_counter()

    configs = load_data("architectures.json")
    resources = load_data("resources.json")
    labels = load_data("development_labels.json")
    splits = load_data("splits.json")

    ids = splits["development"]
    widths = [configs[key]["d_model"] for key in ids]
    parameters = [resources[key]["total"] for key in ids]
    quality = [-labels[key]["valid_ppl"] for key in ids]

    scorers = {
        "original_nsc": original_nsc,
        "pb_nsc": score,
        "matrix_wise": matrix_wise_score,
    }

    # Scores depend only on architecture and can be computed once.
    scores = {
        name: [
            canonical(scorer(configs[key]))
            for key in ids
        ]
        for name, scorer in scorers.items()
    }

    strata = [
        [
            i for i, width in enumerate(widths)
            if width == target_width
        ]
        for target_width in sorted(set(widths))
    ]

    def evaluate_sample(indices):
        sampled_quality = [quality[i] for i in indices]
        quality_ranks = ranks(sampled_quality)

        # Resampling duplicates an architecture, not its training run.
        # Exclude comparisons of copies of the same architecture.
        pairs = [
            (a, b)
            for a in range(len(indices))
            for b in range(a)
            if indices[a] != indices[b]
        ]

        matched_pairs = [
            (a, b)
            for a, b in pairs
            if widths[indices[a]] == widths[indices[b]]
            and (
                min(
                    parameters[indices[a]],
                    parameters[indices[b]],
                )
                / max(
                    parameters[indices[a]],
                    parameters[indices[b]],
                )
                >= 0.90
            )
        ]

        result = {}

        for name, method_scores in scores.items():
            sampled_scores = [
                method_scores[i] for i in indices
            ]

            result[name] = {
                "spearman_rho": correlation(
                    ranks(sampled_scores), quality_ranks
                ),
                "tau_b": pair_summary(
                    sampled_scores, sampled_quality, pairs
                )["tau_b"],
                "controlled_tau_b": pair_summary(
                    sampled_scores,
                    sampled_quality,
                    matched_pairs,
                )["tau_b"],
            }

        return result

    comparisons = {
        "pb_minus_nsc": ("pb_nsc", "original_nsc"),
        "pb_minus_matrix_wise": ("pb_nsc", "matrix_wise"),
    }

    original = evaluate_sample(list(range(len(ids))))

    differences = {
        comparison: {metric: [] for metric in METRICS}
        for comparison in comparisons
    }

    rng = random.Random(SEED)

    for repetition in range(REPEATS):
        indices = [
            rng.choice(group)
            for group in strata
            for _ in range(len(group))
        ]

        result = evaluate_sample(indices)

        for comparison, (left, right) in comparisons.items():
            for metric in METRICS:
                left_value = result[left][metric]
                right_value = result[right][metric]

                if left_value is not None and right_value is not None:
                    differences[comparison][metric].append(
                        left_value - right_value
                    )

        if (repetition + 1) % 200 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Completed {repetition + 1}/{REPEATS}; "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    summaries = {}

    for comparison, (left, right) in comparisons.items():
        summaries[comparison] = {}
        print(f"\n{comparison}")

        for metric in METRICS:
            values = differences[comparison][metric]
            observed = original[left][metric] - original[right][metric]

            if values:
                lower = percentile(values, 0.025)
                upper = percentile(values, 0.975)
            else:
                lower = None
                upper = None

            summaries[comparison][metric] = {
                "observed_difference": observed,
                "valid_resamples": len(values),
                "invalid_resamples": REPEATS - len(values),
                "percentile_95_interval": [lower, upper],
            }

            if values:
                print(
                    f"  {metric}: delta={observed:.6f}, "
                    f"95% interval=[{lower:.6f}, {upper:.6f}], "
                    f"valid={len(values)}"
                )
            else:
                print(f"  {metric}: no valid resamples")

    output = {
        "split": "development",
        "bootstrap_unit": "architecture",
        "sampling": (
            "Paired sampling with replacement within hidden-width "
            "strata, preserving each stratum's original size."
        ),
        "controlled_pairs": (
            "Rebuilt in every resample: distinct original architectures, "
            "same hidden width, parameter min/max >= 0.90. "
            "Repeated architectures retain sampling multiplicity."
        ),
        "interval_method": "95% percentile bootstrap",
        "seed": SEED,
        "repeats": REPEATS,
        "original_metrics": original,
        "comparisons": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": (
            "Exploratory uncertainty conditional on this archive and its "
            "width composition; not training-seed uncertainty, "
            "causal evidence or independent-benchmark validation."
        ),
    }

    output_path = ROOT / "bootstrap_development_results.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path.name}")


if __name__ == "__main__":
    main()