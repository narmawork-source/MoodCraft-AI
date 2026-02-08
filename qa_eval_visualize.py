#!/usr/bin/env python3
"""Visualize QAEval results from CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create charts from QAEval results.")
    parser.add_argument("--input", default="analysis_output/qa_eval_results.csv", help="Path to QAEval results CSV")
    parser.add_argument("--output-dir", default="analysis_output", help="Directory to write charts")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_config = out_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    import matplotlib.pyplot as plt

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    if "qaeval_grade" not in df.columns:
        raise ValueError(
            "Column 'qaeval_grade' not found. Run qa_eval.py scoring first to generate graded results."
        )

    grades = df["qaeval_grade"].fillna("UNKNOWN").str.upper()
    counts = grades.value_counts().reindex(["CORRECT", "INCORRECT", "UNKNOWN"], fill_value=0)
    total = len(df)
    accuracy = (counts["CORRECT"] / total * 100) if total else 0.0

    # Chart 1: Grade distribution bar
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("QAEval Grade Distribution")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02 * max(1, counts.values.max()), str(int(v)), ha="center")
    fig.tight_layout()
    bar_path = out_dir / "qa_eval_grade_distribution.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    # Chart 2: Accuracy donut
    fig, ax = plt.subplots(figsize=(6, 6))
    vals = [counts["CORRECT"], total - counts["CORRECT"]]
    labels = ["Correct", "Not Correct"]
    wedges, _ = ax.pie(vals, labels=labels, startangle=90, colors=["#2ca02c", "#d62728"], wedgeprops={"width": 0.45})
    ax.set_title(f"QAEval Accuracy: {accuracy:.2f}%")
    fig.tight_layout()
    donut_path = out_dir / "qa_eval_accuracy_donut.png"
    fig.savefig(donut_path, dpi=150)
    plt.close(fig)

    # Chart 3: Per-question grade (horizontal)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * total)))
    y_labels = [f"Q{i+1}" for i in range(total)]
    score = grades.map({"CORRECT": 1, "INCORRECT": 0}).fillna(-1)
    bar_colors = score.map({1: "#2ca02c", 0: "#d62728", -1: "#7f7f7f"}).tolist()
    ax.barh(y_labels, score, color=bar_colors)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["Unknown", "Incorrect", "Correct"])
    ax.set_title("Per-Question QAEval Outcome")
    ax.set_xlabel("Outcome")
    fig.tight_layout()
    per_q_path = out_dir / "qa_eval_per_question.png"
    fig.savefig(per_q_path, dpi=150)
    plt.close(fig)

    print(f"Input: {in_path}")
    print(f"Rows: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Saved charts:")
    print(f"- {bar_path}")
    print(f"- {donut_path}")
    print(f"- {per_q_path}")


if __name__ == "__main__":
    main()
