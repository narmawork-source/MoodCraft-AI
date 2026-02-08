#!/usr/bin/env python3
"""Generate chart-ready sales analysis outputs from sales_data.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def build_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

    age_bins = [0, 25, 35, 45, 55, 200]
    age_labels = ["<=25", "26-35", "36-45", "46-55", "56+"]
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=age_bins, labels=age_labels, right=True)

    outputs: dict[str, pd.DataFrame] = {}

    outputs["kpi_overview"] = pd.DataFrame(
        {
            "metric": [
                "rows",
                "total_sales",
                "avg_sales",
                "median_sales",
                "std_sales",
                "min_sales",
                "max_sales",
                "avg_satisfaction",
                "median_satisfaction",
                "std_satisfaction",
            ],
            "value": [
                len(df),
                df["Sales"].sum(),
                df["Sales"].mean(),
                df["Sales"].median(),
                df["Sales"].std(),
                df["Sales"].min(),
                df["Sales"].max(),
                df["Customer_Satisfaction"].mean(),
                df["Customer_Satisfaction"].median(),
                df["Customer_Satisfaction"].std(),
            ],
        }
    )

    outputs["monthly_trend"] = (
        df.groupby("YearMonth", as_index=False)
        .agg(
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            orders=("Sales", "size"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("YearMonth")
    )

    outputs["monthly_trend"]["mom_growth_pct"] = (
        outputs["monthly_trend"]["total_sales"].pct_change() * 100
    )

    outputs["quarterly_trend"] = (
        df.groupby("Quarter", as_index=False)
        .agg(total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"), orders=("Sales", "size"))
        .sort_values("Quarter")
    )

    outputs["yearly_trend"] = (
        df.groupby("Year", as_index=False)
        .agg(total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"), orders=("Sales", "size"))
        .sort_values("Year")
    )

    outputs["product_summary"] = (
        df.groupby("Product", as_index=False)
        .agg(
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            std_sales=("Sales", "std"),
            orders=("Sales", "size"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )

    outputs["region_summary"] = (
        df.groupby("Region", as_index=False)
        .agg(
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            std_sales=("Sales", "std"),
            orders=("Sales", "size"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )

    outputs["product_region_matrix"] = (
        df.groupby(["Product", "Region"], as_index=False)
        .agg(total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"), orders=("Sales", "size"))
        .sort_values("total_sales", ascending=False)
    )

    outputs["gender_segmentation"] = (
        df.groupby("Customer_Gender", as_index=False)
        .agg(
            customers=("Sales", "size"),
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            std_sales=("Sales", "std"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )

    outputs["age_segmentation"] = (
        df.groupby("Age_Group", as_index=False, observed=False)
        .agg(
            customers=("Sales", "size"),
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            std_sales=("Sales", "std"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
    )

    outputs["age_gender_segmentation"] = (
        df.groupby(["Age_Group", "Customer_Gender"], as_index=False, observed=False)
        .agg(
            customers=("Sales", "size"),
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )

    outputs["month_of_year_seasonality"] = (
        df.groupby("Month", as_index=False)
        .agg(total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"), orders=("Sales", "size"))
        .sort_values("total_sales", ascending=False)
    )

    return outputs


def maybe_generate_charts(outputs: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    chart_paths: list[Path] = []
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return chart_paths

    monthly = outputs["monthly_trend"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly["YearMonth"], monthly["total_sales"])
    ax.set_title("Monthly Total Sales")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    path = output_dir / "monthly_total_sales.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    product = outputs["product_summary"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(product["Product"], product["total_sales"])
    ax.set_title("Total Sales by Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    path = output_dir / "product_total_sales.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    region = outputs["region_summary"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(region["Region"], region["total_sales"])
    ax.set_title("Total Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    path = output_dir / "region_total_sales.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    # Combined dashboard for a single, at-a-glance visual summary.
    product = outputs["product_summary"]
    region = outputs["region_summary"]
    age_seg = outputs["age_segmentation"]
    gender_seg = outputs["gender_segmentation"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(monthly["YearMonth"], monthly["total_sales"], color="#1f77b4")
    ax1.set_title("Monthly Total Sales")
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("Sales")
    ax1.tick_params(axis="x", rotation=90)

    ax2.bar(product["Product"], product["total_sales"], color="#2ca02c")
    ax2.set_title("Sales by Product")
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Sales")

    ax3.bar(region["Region"], region["total_sales"], color="#ff7f0e")
    ax3.set_title("Sales by Region")
    ax3.set_xlabel("Region")
    ax3.set_ylabel("Sales")

    ax4.bar(age_seg["Age_Group"].astype(str), age_seg["total_sales"], color="#9467bd", alpha=0.75, label="Age")
    if "Customer_Gender" in gender_seg.columns:
        gender_labels = gender_seg["Customer_Gender"].astype(str).tolist()
        gender_values = gender_seg["total_sales"].tolist()
        ax4_twin = ax4.twinx()
        ax4_twin.plot(gender_labels, gender_values, color="#d62728", marker="o", linewidth=2, label="Gender")
        ax4_twin.set_ylabel("Sales (Gender Line)")
    ax4.set_title("Demographic Sales (Age bars + Gender line)")
    ax4.set_xlabel("Age Group")
    ax4.set_ylabel("Sales (Age bars)")

    fig.tight_layout()
    path = output_dir / "sales_dashboard.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    # 1) Sales performance by time period
    quarterly = outputs["quarterly_trend"].copy()
    yearly = outputs["yearly_trend"].copy()
    monthly_growth = monthly.copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    t1, t2, t3, t4 = axes.flatten()

    t1.plot(monthly["YearMonth"], monthly["total_sales"], color="#1f77b4")
    t1.set_title("Monthly Sales Trend")
    t1.set_xlabel("Year-Month")
    t1.set_ylabel("Total Sales")
    t1.tick_params(axis="x", rotation=90)

    t2.bar(quarterly["Quarter"], quarterly["total_sales"], color="#17becf")
    t2.set_title("Quarterly Total Sales")
    t2.set_xlabel("Quarter")
    t2.set_ylabel("Total Sales")
    t2.tick_params(axis="x", rotation=90)

    t3.bar(yearly["Year"].astype(str), yearly["total_sales"], color="#2ca02c")
    t3.set_title("Yearly Total Sales")
    t3.set_xlabel("Year")
    t3.set_ylabel("Total Sales")

    t4.plot(monthly_growth["YearMonth"], monthly_growth["mom_growth_pct"], color="#d62728")
    t4.axhline(0, color="black", linewidth=1)
    t4.set_title("Month-over-Month Growth (%)")
    t4.set_xlabel("Year-Month")
    t4.set_ylabel("Growth %")
    t4.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    path = output_dir / "visual_1_sales_time_period.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    # 2) Product and regional analysis
    prod_region = outputs["product_region_matrix"].copy()
    pivot = prod_region.pivot(index="Product", columns="Region", values="total_sales")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p1, p2, p3 = axes.flatten()

    p1.bar(product["Product"], product["total_sales"], color="#2ca02c")
    p1.set_title("Total Sales by Product")
    p1.set_xlabel("Product")
    p1.set_ylabel("Total Sales")

    p2.bar(region["Region"], region["total_sales"], color="#ff7f0e")
    p2.set_title("Total Sales by Region")
    p2.set_xlabel("Region")
    p2.set_ylabel("Total Sales")

    im = p3.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    p3.set_title("Product-Region Sales Heatmap")
    p3.set_xticks(range(len(pivot.columns)))
    p3.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    p3.set_yticks(range(len(pivot.index)))
    p3.set_yticklabels(list(pivot.index))
    fig.colorbar(im, ax=p3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = output_dir / "visual_2_product_region_analysis.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    # 3) Customer segmentation by demographics
    age_gender = outputs["age_gender_segmentation"].copy()
    age_gender_pivot = age_gender.pivot(
        index="Age_Group", columns="Customer_Gender", values="total_sales"
    ).fillna(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c1, c2, c3 = axes.flatten()

    c1.bar(age_seg["Age_Group"].astype(str), age_seg["total_sales"], color="#9467bd")
    c1.set_title("Sales by Age Group")
    c1.set_xlabel("Age Group")
    c1.set_ylabel("Total Sales")

    c2.bar(gender_seg["Customer_Gender"], gender_seg["total_sales"], color="#8c564b")
    c2.set_title("Sales by Gender")
    c2.set_xlabel("Gender")
    c2.set_ylabel("Total Sales")

    im2 = c3.imshow(age_gender_pivot.values, cmap="OrRd", aspect="auto")
    c3.set_title("Age x Gender Sales Heatmap")
    c3.set_xticks(range(len(age_gender_pivot.columns)))
    c3.set_xticklabels(list(age_gender_pivot.columns))
    c3.set_yticks(range(len(age_gender_pivot.index)))
    c3.set_yticklabels([str(x) for x in age_gender_pivot.index])
    fig.colorbar(im2, ax=c3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = output_dir / "visual_3_customer_segmentation.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    # 4) Statistical measures (median, standard deviation, etc.)
    kpi = outputs["kpi_overview"].copy()
    kpi_map = dict(zip(kpi["metric"], kpi["value"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s1, s2, s3 = axes.flatten()

    prod_stats = product[["Product", "avg_sales", "median_sales", "std_sales"]].set_index("Product")
    prod_stats.plot(kind="bar", ax=s1)
    s1.set_title("Product: Mean vs Median vs Std")
    s1.set_xlabel("Product")
    s1.set_ylabel("Sales")
    s1.tick_params(axis="x", rotation=0)

    reg_stats = region[["Region", "avg_sales", "median_sales", "std_sales"]].set_index("Region")
    reg_stats.plot(kind="bar", ax=s2)
    s2.set_title("Region: Mean vs Median vs Std")
    s2.set_xlabel("Region")
    s2.set_ylabel("Sales")
    s2.tick_params(axis="x", rotation=0)

    s3.axis("off")
    stats_text = (
        f"Overall Sales Stats\\n"
        f"Mean: {kpi_map.get('avg_sales', 0):.2f}\\n"
        f"Median: {kpi_map.get('median_sales', 0):.2f}\\n"
        f"Std Dev: {kpi_map.get('std_sales', 0):.2f}\\n"
        f"Min: {kpi_map.get('min_sales', 0):.0f}\\n"
        f"Max: {kpi_map.get('max_sales', 0):.0f}\\n\\n"
        f"Satisfaction Stats\\n"
        f"Mean: {kpi_map.get('avg_satisfaction', 0):.2f}\\n"
        f"Median: {kpi_map.get('median_satisfaction', 0):.2f}\\n"
        f"Std Dev: {kpi_map.get('std_satisfaction', 0):.2f}"
    )
    s3.text(0.02, 0.98, stats_text, va="top", fontsize=12)
    s3.set_title("Overall Statistical Measures")

    fig.tight_layout()
    path = output_dir / "visual_4_statistical_measures.png"
    fig.savefig(path, dpi=150)
    chart_paths.append(path)
    plt.close(fig)

    return chart_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chart-ready sales summary outputs.")
    parser.add_argument(
        "--input",
        default="/Users/narmasarav/Downloads/sales_data.csv",
        help="Path to input CSV (default: /Users/narmasarav/Downloads/sales_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory to write summary CSV files and optional charts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    outputs = build_outputs(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_csvs: list[Path] = []
    for name, frame in outputs.items():
        out = output_dir / f"{name}.csv"
        frame.to_csv(out, index=False)
        saved_csvs.append(out)

    charts = maybe_generate_charts(outputs, output_dir)

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Saved {len(saved_csvs)} CSV files:")
    for p in saved_csvs:
        print(f"- {p}")

    if charts:
        print(f"Saved {len(charts)} charts:")
        for p in charts:
            print(f"- {p}")
    else:
        print("No charts generated (matplotlib not installed).")


if __name__ == "__main__":
    main()
