"""Inspect ranking changes on development architectures only."""

import csv
import json
import statistics
from pathlib import Path

from evaluate import canonical
from nsc import layers, original_nsc
from pb_nsc_score import score


ROOT = Path(__file__).resolve().parent


def load_data(filename):
    return json.loads(
        (ROOT / "data" / filename).read_text(encoding="utf-8")
    )


def architecture_info(arch_id, configs, resources, labels):
    config = configs[arch_id]
    d, heads, inner = layers(config)

    return {
        "id": arch_id,
        "width": d,
        "depth": config["n_layer"],
        "parameters": resources[arch_id]["total"],
        "ppl": labels[arch_id]["valid_ppl"],
        "mean_heads": statistics.mean(heads),
        "mean_ffn_ratio": statistics.mean(f / d for f in inner),
        "nsc": canonical(original_nsc(config)),
        "pb_nsc": canonical(score(config)),
    }


def compare(a, b):
    return (a > b) - (a < b)


def main():
    configs = load_data("architectures.json")
    resources = load_data("resources.json")
    labels = load_data("development_labels.json")
    splits = load_data("splits.json")

    architectures = [
        architecture_info(arch_id, configs, resources, labels)
        for arch_id in splits["development"]
    ]

    rows = []
    skipped_ties = 0

    for i, a in enumerate(architectures):
        for b in architectures[:i]:
            old_order = compare(a["nsc"], b["nsc"])
            new_order = compare(a["pb_nsc"], b["pb_nsc"])

            if old_order == new_order:
                continue

            # Treat ties separately rather than calling them reversals.
            if (
                old_order == 0
                or new_order == 0
                or a["ppl"] == b["ppl"]
            ):
                skipped_ties += 1
                continue

            # "Promoted" means preferred by PB-NSC over NSC.
            promoted, demoted = (
                (a, b) if new_order > 0 else (b, a)
            )

            same_width = a["width"] == b["width"]
            parameter_ratio = (
                min(a["parameters"], b["parameters"])
                / max(a["parameters"], b["parameters"])
            )

            row = {
                "change": (
                    "corrected"
                    if promoted["ppl"] < demoted["ppl"]
                    else "worsened"
                ),
                "same_width": same_width,
                "size_controlled": (
                    same_width and parameter_ratio >= 0.90
                ),
                "parameter_ratio": parameter_ratio,
            }

            for prefix, info in [
                ("promoted", promoted),
                ("demoted", demoted),
            ]:
                for key, value in info.items():
                    row[f"{prefix}_{key}"] = value

            rows.append(row)

    fields = [
        "change",
        "same_width",
        "size_controlled",
        "parameter_ratio",
    ] + [
        f"{prefix}_{key}"
        for prefix in ["promoted", "demoted"]
        for key in [
            "id", "width", "depth", "parameters", "ppl",
            "mean_heads", "mean_ffn_ratio", "nsc", "pb_nsc",
        ]
    ]

    csv_path = ROOT / "pair_changes_development.csv"
    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    subsets = {
        "all_changed_pairs": rows,
        "same_width_changed_pairs": [
            row for row in rows if row["same_width"]
        ],
        "different_width_changed_pairs": [
            row for row in rows if not row["same_width"]
        ],
        "size_controlled_changed_pairs": [
            row for row in rows if row["size_controlled"]
        ],
    }

    summaries = {}

    for name, subset in subsets.items():
        corrected = sum(
            row["change"] == "corrected" for row in subset
        )
        worsened = sum(
            row["change"] == "worsened" for row in subset
        )

        summary = {
            "changed_pairs": len(subset),
            "corrected": corrected,
            "worsened": worsened,
            "net_corrected": corrected - worsened,
            "patterns": {},
        }

        # Describe structural preferences separately for
        # successful and unsuccessful ranking changes.
        for outcome in ["corrected", "worsened"]:
            selected = [
                row for row in subset
                if row["change"] == outcome
            ]

            patterns = {"pairs": len(selected)}

            for feature in ["mean_heads", "mean_ffn_ratio"]:
                differences = [
                    row[f"promoted_{feature}"]
                    - row[f"demoted_{feature}"]
                    for row in selected
                ]

                patterns[feature] = {
                    "promoted_lower": sum(
                        value < 0 for value in differences
                    ),
                    "equal": sum(
                        value == 0 for value in differences
                    ),
                    "promoted_higher": sum(
                        value > 0 for value in differences
                    ),
                }

            summary["patterns"][outcome] = patterns

        summaries[name] = summary

        print(
            f"\n{name}: changed={len(subset)}, "
            f"corrected={corrected}, worsened={worsened}, "
            f"net={corrected - worsened}"
        )

        for outcome, patterns in summary["patterns"].items():
            print(f"  {outcome}:")
            for feature in ["mean_heads", "mean_ffn_ratio"]:
                counts = patterns[feature]
                print(
                    f"    {feature}: "
                    f"lower={counts['promoted_lower']}, "
                    f"equal={counts['equal']}, "
                    f"higher={counts['promoted_higher']}"
                )

    output = {
        "split": "development",
        "note": (
            "Descriptive counts over overlapping changed pairs; "
            "not independent samples or causal evidence. "
            "Layer means summarize heterogeneous configurations."
        ),
        "skipped_changed_pairs_with_ties": skipped_ties,
        "subsets": summaries,
    }

    json_path = ROOT / "pair_changes_summary_development.json"
    json_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {csv_path.name}")
    print(f"Saved: {json_path.name}")
    print(f"Changed pairs skipped because of ties: {skipped_ties}")


if __name__ == "__main__":
    main()