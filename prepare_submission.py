"""Organize saved results and documentation; never run evaluation."""

import csv
import io
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREEZE = "04fd56967323d82eb9bce081e73b7e29a4045d03"
MILESTONE = "2878ddddd08c90af26f9493608105673d03b8bd2"


def load_result(name, split):
    data = json.loads(
        (ROOT / name).read_text(encoding="utf-8")
    )
    if data.get("split") != split or data.get("architectures") != 100:
        raise ValueError(f"Unexpected split or architecture count: {name}")
    return data


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main():
    ranking_rows = []
    selection_rows = []
    bootstrap_rows = []

    for split in ["development", "final"]:
        base = load_result(f"pb_nsc_{split}.json", split)

        for method, result in base["ranking"].items():
            if method == "ablation":
                continue
            name = "pb_nsc" if method == "student" else method

            ranking_rows.append({
                "split": split,
                "method": name,
                "spearman_rho": result["spearman_rho"],
                "kendall_tau_b": result["all_pairs"]["tau_b"],
                "controlled_tau_b": result["size_controlled_pairs"]["tau_b"],
                "controlled_pairs": result["size_controlled_pairs"]["pairs"],
            })

        evaluations = [(None, base)]

        for method in ["attention_only", "ffn_only", "matrix_wise"]:
            data = load_result(f"{method}_{split}.json", split)

            if (
                data["ranking"]["original_nsc"]
                != base["ranking"]["original_nsc"]
            ):
                raise ValueError(f"Baseline mismatch: {method}, {split}")

            result = data["ranking"]["student"]
            ranking_rows.append({
                "split": split,
                "method": method,
                "spearman_rho": result["spearman_rho"],
                "kendall_tau_b": result["all_pairs"]["tau_b"],
                "controlled_tau_b": result["size_controlled_pairs"]["tau_b"],
                "controlled_pairs": result["size_controlled_pairs"]["pairs"],
            })
            evaluations.append((method, data))

        for variant, data in evaluations:
            if len(data["selection"]) != 8:
                raise ValueError("Expected eight budget/k settings.")

            for setting in data["selection"]:
                for method, result in setting["methods"].items():
                    if method == "ablation":
                        continue
                    if variant and method != "student":
                        continue

                    name = variant or (
                        "pb_nsc" if method == "student" else method
                    )

                    selection_rows.append({
                        "split": split,
                        "method": name,
                        "parameter_cap": setting["parameter_cap"],
                        "k": setting["k"],
                        "eligible": setting["eligible"],
                        "oracle_ppl": setting["oracle_ppl"],
                        "best_ppl_mean": result["mean_best_ppl"],
                        "regret_mean": result["mean_regret"],
                        "best_ppl_sd": result["sd_best_ppl"],
                        "trials": result["trials"],
                    })

        boot = json.loads(
            (ROOT / f"bootstrap_{split}_results.json").read_text(
                encoding="utf-8"
            )
        )

        if (
            boot["split"] != split
            or boot["repeats"] != 2000
            or boot["seed"] != 5491
        ):
            raise ValueError(f"Unexpected bootstrap settings: {split}")

        for comparison, metrics in boot["comparisons"].items():
            for metric, result in metrics.items():
                bootstrap_rows.append({
                    "split": split,
                    "comparison": comparison,
                    "metric": metric,
                    "difference": result["observed_difference"],
                    "lower_95": result["percentile_95_interval"][0],
                    "upper_95": result["percentile_95_interval"][1],
                    "valid_resamples": result["valid_resamples"],
                    "invalid_resamples": result["invalid_resamples"],
                })

    readme = (ROOT / "README_PB_NSC.md").read_text(encoding="utf-8")
    start = readme.index("## Final evaluation status")
    end = readme.index("## References", start)

    section = f"""## Final evaluation status and protocol

Final evaluation is complete. Methods were frozen before evaluating final
outcomes at commit `{FREEZE}`.
See [FINAL_PROTOCOL.md](FINAL_PROTOCOL.md) for the historical protocol;
its pre-evaluation status text is retained as a historical record.
See [FINAL_RESULTS.md](FINAL_RESULTS.md) for the numerical summary.

### Reproduce final results

```sh
python evaluate.py --split final --student pb_nsc_score.py --output pb_nsc_final.json
python evaluate.py --split final --student score_attention_only.py --output attention_only_final.json
python evaluate.py --split final --student score_ffn_only.py --output ffn_only_final.json
python evaluate.py --split final --student score_matrix_wise.py --output matrix_wise_final.json
python analyze_width.py --split final
python analyze_pair_changes.py --split final
python bootstrap_development.py --split final
```

These commands reproduce frozen analyses, without retuning. The bootstrap
filename is historical; its --split option selects the dataset.
Most commands take seconds; final bootstrap took about 13 seconds locally.

### Regenerate summary tables without running experiments

```sh
python prepare_submission.py
```

This standard-library script reads saved evaluation JSON, backs up files
before replacing them, updates documentation and exports three CSV tables
under reports/. It does not call evaluation, read label files, change
scoring code, or upload anything. Existing backups remain local.

The pre-final freeze differs from the eventual graded final commit,
which must also contain final results and submission materials.

"""

    outputs = {
        "README_PB_NSC.md": readme[:start] + section + readme[end:],
        "reports/ranking_summary.csv": csv_text(ranking_rows),
        "reports/selection_summary.csv": csv_text(selection_rows),
        "reports/bootstrap_summary.csv": csv_text(bootstrap_rows),
        "FREEZE_RECORD.md": f"""# Evaluation version record

Milestone commit: `{MILESTONE}`

Pre-final code and protocol freeze: `{FREEZE}`

Final evaluation was subsequently completed using the frozen methods.
The graded final commit must be recorded after final results and report
are uploaded. It is not the pre-final freeze commit.
""",
    }

    final_rows = [
        row for row in ranking_rows if row["split"] == "final"
    ]

    summary = "# Final Results Summary\n\n"
    summary += (
        "| Method | Spearman rho | Kendall tau-b | Controlled tau-b |\n"
        "|---|---:|---:|---:|\n"
    )

    for row in final_rows:
        summary += (
            f"| {row['method']} | {row['spearman_rho']:.6f} | "
            f"{row['kendall_tau_b']:.6f} | "
            f"{row['controlled_tau_b']:.6f} |\n"
        )

    summary += """
## Interpretation

PB-NSC improves overall ranking on this fixed archive. Overall Spearman
and Kendall differences versus NSC have positive final bootstrap intervals.
The controlled improvement interval includes zero; geometric aggregation
has no established distinct advantage over matrix-wise weighting.

PB-NSC matches NSC's best-selected PPL in all eight final budget/k settings.
The width-512 controlled subset has negative tau-b for both methods.
Most ranking improvement involves cross-width comparisons; FFN shape
preferences are an important descriptive pattern, not causal evidence.

Development uses validation PPL and final uses test PPL on different
architectures. Do not directly compare absolute PPL across splits.
Bootstrap uncertainty is not training-seed uncertainty.

Complete derived tables are in reports/. Their source is the saved JSON
and they can be regenerated with python prepare_submission.py.
"""
    outputs["FINAL_RESULTS.md"] = summary

    # Nothing above this line modifies project files.
    backup = (
        ROOT / "local_backups"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    # Back up every existing file that will be replaced.
    for name in outputs:
        existing = ROOT / name
        if existing.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, target)

    for name, content in outputs.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoding = "utf-8-sig" if target.suffix == ".csv" else "utf-8"
        target.write_text(content, encoding=encoding)
        print(f"SAVED: {name}")

    print(f"Backup folder (if files existed): {backup}")
    print(
        "DONE: no evaluation was run; "
        "scoring code and data were not changed."
    )


if __name__ == "__main__":
    main()