"""Compare development rankings within each hidden-width group."""

import argparse
import json
from pathlib import Path

from evaluate import canonical, correlation, pair_summary, ranks
from nsc import original_nsc
from pb_nsc_score import (
    attention_only_score,
    ffn_only_score,
    matrix_wise_score,
    score,
)


ROOT = Path(__file__).resolve().parent

METHODS = {
    "original_nsc": original_nsc,
    "attention_only": attention_only_score,
    "ffn_only": ffn_only_score,
    "matrix_wise": matrix_wise_score,
    "pb_nsc": score,
}


def load_data(filename):
    return json.loads(
        (ROOT / "data" / filename).read_text(encoding="utf-8")
    )


def main(split):
    configs = load_data("architectures.json")
    resources = load_data("resources.json")
    splits = load_data("splits.json")
    labels = load_data(f"{split}_labels.json")
    label_key = "valid_ppl" if split == "development" else "test_ppl"

    selected_ids = splits[split]
    widths = sorted({
        configs[arch_id]["d_model"]
        for arch_id in selected_ids
    })

    results = []

    for width in widths:
        ids = [
            arch_id
            for arch_id in selected_ids
            if configs[arch_id]["d_model"] == width
        ]

        # Larger negative PPL means better performance.
        quality = [
            -labels[arch_id][label_key]
            for arch_id in ids
        ]

        parameters = [
            resources[arch_id]["total"]
            for arch_id in ids
        ]

        all_pairs = [
            (i, j)
            for i in range(len(ids))
            for j in range(i)
        ]

        # Width is already fixed within this group.
        matched_pairs = [
            (i, j)
            for i, j in all_pairs
            if min(parameters[i], parameters[j])
            / max(parameters[i], parameters[j]) >= 0.90
        ]

        group = {
            "hidden_width": width,
            "architectures": len(ids),
            "all_pair_count": len(all_pairs),
            "size_controlled_pair_count": len(matched_pairs),
            "methods": {},
        }

        print(
            f"\nWidth={width}, architectures={len(ids)}, "
            f"controlled_pairs={len(matched_pairs)}"
        )

        for name, scorer in METHODS.items():
            scores = [
                canonical(scorer(configs[arch_id]))
                for arch_id in ids
            ]

            rho = correlation(ranks(scores), ranks(quality))
            all_summary = pair_summary(
                scores, quality, all_pairs
            )
            controlled_summary = pair_summary(
                scores, quality, matched_pairs
            )

            group["methods"][name] = {
                "spearman_rho": rho,
                "all_pairs": all_summary,
                "size_controlled_pairs": controlled_summary,
            }

            def display(value):
                return "NA" if value is None else f"{value:.6f}"

            print(
                f"{name:16s} "
                f"rho={display(rho)} "
                f"tau={display(all_summary['tau_b'])} "
                f"controlled_tau="
                f"{display(controlled_summary['tau_b'])}"
            )

        results.append(group)

    output = {
        "split": split,
        "analysis": "within-hidden-width ranking",
        "groups": results,
    }

    output_path = ROOT / f"width_analysis_{split}.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze architecture rankings within hidden-width groups."
    )
    parser.add_argument(
        "--split",
        choices=["development", "final"],
        default="development",
        help="Dataset split; defaults to development.",
    )
    args = parser.parse_args()
    main(args.split)