"""Plot saved final results; does not run evaluation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPORTS = Path(__file__).resolve().parent
PROJECT = REPORTS.parent


def main():
    data = json.loads(
        (PROJECT / "width_analysis_final.json").read_text(
            encoding="utf-8"
        )
    )

    if data["split"] != "final":
        raise ValueError("Expected final width-analysis results.")

    groups = sorted(
        data["groups"],
        key=lambda group: group["hidden_width"],
    )

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(
        figsize=(6.2, 2.8),
        layout="constrained",
    )

    positions = list(range(len(groups)))

    for offset, method, label, color in [
        (-0.17, "original_nsc", "Original NSC", "#526B83"),
        (0.17, "pb_nsc", "PB-NSC", "#C16B32"),
    ]:
        values = [
            group["methods"][method]["size_controlled_pairs"]["tau_b"]
            for group in groups
        ]

        if any(value is None for value in values):
            raise ValueError("A controlled tau-b value is undefined.")

        ax.bar(
            [position + offset for position in positions],
            values,
            width=0.32,
            label=label,
            color=color,
        )

    labels = [
        f"{group['hidden_width']}\n"
        f"({group['size_controlled_pair_count']} pairs)"
        for group in groups
    ]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hidden width (final split)")
    ax.set_ylabel("Controlled Kendall tau-b")
    ax.set_ylim(-0.45, 1.15)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=11,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for extension in ["png", "pdf"]:
        output = REPORTS / f"final_width_control.{extension}"
        fig.savefig(output, dpi=300)
        print(f"SAVED: {output.name}")

    plt.close(fig)
    print("DONE: figure generated from saved results.")


if __name__ == "__main__":
    main()