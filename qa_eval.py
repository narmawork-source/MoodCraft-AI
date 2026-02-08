#!/usr/bin/env python3
"""Evaluate QA performance with LangChain QAEvalChain."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from sales_analysis import build_outputs


def local_answer(question: str, df: pd.DataFrame, outputs: Dict[str, pd.DataFrame]) -> str:
    q = question.lower()

    if any(x in q for x in ["last month", "latest month"]) and "product" in q:
        work = df.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"])
        last_month = work["Date"].dt.to_period("M").max()
        month_df = work[work["Date"].dt.to_period("M") == last_month]
        product_sales = (
            month_df.groupby("Product", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
            .reset_index(drop=True)
        )
        top_row = product_sales.iloc[0]
        return f"Top product in latest month {last_month}: {top_row['Product']} ({int(top_row['Sales'])})"

    if any(x in q for x in ["last quarter", "latest quarter"]) and "product" in q:
        work = df.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"])
        last_quarter = work["Date"].dt.to_period("Q").max()
        qtr_df = work[work["Date"].dt.to_period("Q") == last_quarter]
        product_sales = (
            qtr_df.groupby("Product", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
            .reset_index(drop=True)
        )
        top_row = product_sales.iloc[0]
        return f"Top product in latest quarter {last_quarter}: {top_row['Product']} ({int(top_row['Sales'])})"

    monthly = outputs["monthly_trend"].sort_values("total_sales", ascending=False).iloc[0]
    yearly = outputs["yearly_trend"].sort_values("total_sales", ascending=False).iloc[0]
    product = outputs["product_summary"].iloc[0]
    region = outputs["region_summary"].iloc[0]
    kpi_map = dict(zip(outputs["kpi_overview"]["metric"], outputs["kpi_overview"]["value"]))

    if "best year" in q:
        return f"Best year: {int(yearly['Year'])} ({int(yearly['total_sales'])})"
    if "top month" in q:
        return f"Top month: {monthly['YearMonth']} ({int(monthly['total_sales'])})"
    if "top product" in q:
        return f"Top product: {product['Product']} ({int(product['total_sales'])})"
    if "top region" in q:
        return f"Top region: {region['Region']} ({int(region['total_sales'])})"
    if "median sales" in q:
        return f"Median sales: {kpi_map.get('median_sales', 0):.2f}"
    if "standard deviation" in q or "std" in q:
        return f"Sales std: {kpi_map.get('std_sales', 0):.2f}"

    return (
        f"Summary: top month {monthly['YearMonth']}, best year {int(yearly['Year'])}, "
        f"top product {product['Product']}, top region {region['Region']}"
    )


def create_default_eval_set(df: pd.DataFrame, outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    questions = [
        "What is the top product overall?",
        "Which region has the highest sales?",
        "What is the best year by total sales?",
        "What is the top month by sales?",
        "What is the median sales value?",
        "What is the standard deviation of sales?",
        "Which product performed best in the latest month?",
        "Which product performed best in the latest quarter?",
    ]
    rows: List[Dict[str, str]] = []
    for q in questions:
        gt = local_answer(q, df, outputs)
        rows.append({"question": q, "ground_truth": gt})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answers with QAEvalChain.")
    parser.add_argument("--data-csv", default="/Users/narmasarav/Downloads/sales_data.csv")
    parser.add_argument("--eval-set", default="analysis_output/qa_eval_set.csv")
    parser.add_argument("--out-csv", default="analysis_output/qa_eval_results.csv")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Evaluator model for QAEvalChain",
    )
    parser.add_argument(
        "--build-default-eval-set",
        action="store_true",
        help="Generate a starter eval set before running evaluation.",
    )
    parser.add_argument(
        "--use-local-predictions",
        action="store_true",
        help="Generate predictions via local deterministic logic if prediction column is absent.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build/read eval data and predictions, then exit without QAEvalChain scoring.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    data_path = Path(args.data_csv)
    eval_path = Path(args.eval_set)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    outputs = build_outputs(df)

    if args.build_default_eval_set or not eval_path.exists():
        eval_df = create_default_eval_set(df, outputs)
        eval_df.to_csv(eval_path, index=False)
        print(f"Created eval set: {eval_path}")
    else:
        eval_df = pd.read_csv(eval_path)

    required = {"question", "ground_truth"}
    if not required.issubset(eval_df.columns):
        raise ValueError("Eval set must contain columns: question, ground_truth")

    if "prediction" not in eval_df.columns:
        if args.use_local_predictions:
            eval_df["prediction"] = [local_answer(q, df, outputs) for q in eval_df["question"]]
        else:
            raise ValueError(
                "Eval set is missing prediction column. Add it or rerun with --use-local-predictions."
            )

    if args.prepare_only:
        eval_df.to_csv(eval_path, index=False)
        print(f"Prepared eval set (no scoring): {eval_path}")
        print(f"Rows prepared: {len(eval_df)}")
        return

    if not api_key or "PASTE_REAL_KEY_HERE" in api_key or "your_key_here" in api_key:
        raise ValueError("Set a valid OPENAI_API_KEY in .env before running QAEvalChain scoring.")

    try:
        from langchain_classic.evaluation.qa import QAEvalChain
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise ImportError(
            "Install langchain-classic + langchain-openai to run evaluation: pip install langchain-classic langchain-openai"
        ) from exc

    llm = ChatOpenAI(model=args.model, temperature=0, api_key=api_key)
    chain = QAEvalChain.from_llm(llm)

    examples = [
        {"query": row.question, "answer": row.ground_truth}
        for row in eval_df.itertuples(index=False)
    ]
    predictions = [{"result": row.prediction} for row in eval_df.itertuples(index=False)]

    graded = chain.evaluate(
        examples=examples,
        predictions=predictions,
        question_key="query",
        prediction_key="result",
        answer_key="answer",
    )

    grade_labels: List[str] = []
    reasons: List[str] = []
    for item in graded:
        text = (item.get("text") or str(item)).strip()
        upper = text.upper()
        label = "INCORRECT"
        if "CORRECT" in upper and "INCORRECT" not in upper:
            label = "CORRECT"
        elif "INCORRECT" in upper:
            label = "INCORRECT"
        grade_labels.append(label)
        reasons.append(text)

    eval_df["qaeval_grade"] = grade_labels
    eval_df["qaeval_reason"] = reasons

    accuracy = (eval_df["qaeval_grade"] == "CORRECT").mean() * 100
    eval_df.to_csv(out_path, index=False)

    print(f"Eval set: {eval_path}")
    print(f"Results: {out_path}")
    print(f"Rows evaluated: {len(eval_df)}")
    print(f"QAEvalChain accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
