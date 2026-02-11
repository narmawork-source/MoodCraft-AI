import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from sales_analysis import build_outputs


load_dotenv()

st.set_page_config(page_title="Sales RAG Analyst", page_icon="📊", layout="wide")
st.title("Sales RAG Analyst")
st.caption("Data summary, retrieval, prompt chaining, and memory-enabled Q&A")

DEFAULT_CSV = "/Users/narmasarav/Downloads/sales_data.csv"
SECTION_IDS = ["time", "product_region", "demographics", "statistics"]


def _fmt_num(value: float) -> str:
    return f"{value:,.2f}"


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def build_kb(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, str]]:
    df = load_dataset(csv_path)
    outputs = build_outputs(df)

    monthly = outputs["monthly_trend"]
    yearly = outputs["yearly_trend"]
    product = outputs["product_summary"]
    region = outputs["region_summary"]
    gender = outputs["gender_segmentation"]
    age = outputs["age_segmentation"]
    kpi = outputs["kpi_overview"]

    top_month = monthly.sort_values("total_sales", ascending=False).iloc[0]
    bottom_month = monthly.sort_values("total_sales", ascending=True).iloc[0]
    best_year = yearly.sort_values("total_sales", ascending=False).iloc[0]

    top_product = product.iloc[0]
    low_product = product.iloc[-1]
    top_region = region.iloc[0]

    top_gender = gender.sort_values("total_sales", ascending=False).iloc[0]
    top_age = age.sort_values("total_sales", ascending=False).iloc[0]

    kpi_map = dict(zip(kpi["metric"], kpi["value"]))

    kb_text = {
        "time": (
            "Sales performance by time period: "
            f"Top month={top_month['YearMonth']} ({_fmt_num(top_month['total_sales'])}), "
            f"Lowest month={bottom_month['YearMonth']} ({_fmt_num(bottom_month['total_sales'])}), "
            f"Best year={int(best_year['Year'])} ({_fmt_num(best_year['total_sales'])})."
        ),
        "product_region": (
            "Product and regional analysis: "
            f"Top product={top_product['Product']} ({_fmt_num(top_product['total_sales'])}), "
            f"Lowest product={low_product['Product']} ({_fmt_num(low_product['total_sales'])}), "
            f"Top region={top_region['Region']} ({_fmt_num(top_region['total_sales'])})."
        ),
        "demographics": (
            "Customer segmentation by demographics: "
            f"Top gender by sales={top_gender['Customer_Gender']} ({_fmt_num(top_gender['total_sales'])}), "
            f"Top age group by sales={top_age['Age_Group']} ({_fmt_num(top_age['total_sales'])})."
        ),
        "statistics": (
            "Statistical measures: "
            f"Mean sales={_fmt_num(kpi_map.get('avg_sales', 0))}, "
            f"Median sales={_fmt_num(kpi_map.get('median_sales', 0))}, "
            f"Std sales={_fmt_num(kpi_map.get('std_sales', 0))}, "
            f"Mean satisfaction={_fmt_num(kpi_map.get('avg_satisfaction', 0))}, "
            f"Median satisfaction={_fmt_num(kpi_map.get('median_satisfaction', 0))}, "
            f"Std satisfaction={_fmt_num(kpi_map.get('std_satisfaction', 0))}."
        ),
    }

    return df, outputs, kb_text


def keyword_retriever(question: str) -> List[str]:
    q = question.lower()
    selected = set()

    if any(w in q for w in ["time", "month", "quarter", "year", "trend", "growth", "season"]):
        selected.add("time")
    if any(w in q for w in ["product", "region", "place", "area", "west", "east", "north", "south"]):
        selected.add("product_region")
    if any(w in q for w in ["customer", "buyer", "demographic", "gender", "age", "segment"]):
        selected.add("demographics")
    if any(w in q for w in ["median", "mean", "std", "standard deviation", "stat", "distribution"]):
        selected.add("statistics")

    if not selected:
        selected = set(SECTION_IDS)

    return sorted(selected)


def llm_section_selector(client: OpenAI, question: str, memory_text: str) -> List[str]:
    planner_prompt = (
        "You are a query planner for sales analytics. "
        "From these sections only: time, product_region, demographics, statistics. "
        "Return a comma-separated list of relevant section ids only, no extra words."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": f"Memory:\n{memory_text}\n\nQuestion:\n{question}"},
        ],
    )

    raw = (resp.choices[0].message.content or "").lower()
    selected = [sid for sid in SECTION_IDS if sid in raw]
    return selected or keyword_retriever(question)


def answer_with_rag(
    client: OpenAI,
    question: str,
    selected_sections: List[str],
    kb_text: Dict[str, str],
    memory_messages: List[Dict[str, str]],
) -> str:
    context = "\n".join([f"[{sid}] {kb_text[sid]}" for sid in selected_sections])
    memory_block = "\n".join([f"{m['role']}: {m['content']}" for m in memory_messages[-6:]])

    system_prompt = (
        "You are a sales analytics assistant. Use only the retrieved context for numeric claims. "
        "If data is not in context, say it is not available. Be concise and structured."
    )

    user_prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Conversation memory:\n{memory_block or 'No prior messages.'}\n\n"
        f"User question:\n{question}\n\n"
        "Return:\n"
        "1) Direct answer\n"
        "2) Supporting metrics\n"
        "3) Suggested follow-up analysis"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or "No response generated."


def local_fallback_answer(question: str, df: pd.DataFrame, outputs: Dict[str, pd.DataFrame]) -> str:
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
        ranking = "\n".join(
            [
                f"{idx + 1}. {row['Product']} - {int(row['Sales']):,}"
                for idx, (_, row) in enumerate(product_sales.iterrows())
            ]
        )
        return (
            f"Top product in the latest month ({last_month}): **{top_row['Product']}** with **{int(top_row['Sales']):,}** sales.\n\n"
            f"Monthly product ranking:\n{ranking}"
        )

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
        ranking = "\n".join(
            [
                f"{idx + 1}. {row['Product']} - {int(row['Sales']):,}"
                for idx, (_, row) in enumerate(product_sales.iterrows())
            ]
        )
        return (
            f"Top product in the latest quarter ({last_quarter}): **{top_row['Product']}** with **{int(top_row['Sales']):,}** sales.\n\n"
            f"Quarterly product ranking:\n{ranking}"
        )

    parts = []
    if "time" in keyword_retriever(question):
        monthly = outputs["monthly_trend"].sort_values("total_sales", ascending=False).iloc[0]
        yearly = outputs["yearly_trend"].sort_values("total_sales", ascending=False).iloc[0]
        parts.append(
            f"Time: top month is {monthly['YearMonth']} ({_fmt_num(monthly['total_sales'])}), best year is {int(yearly['Year'])} ({_fmt_num(yearly['total_sales'])})."
        )

    if "product_region" in keyword_retriever(question):
        product = outputs["product_summary"].iloc[0]
        region = outputs["region_summary"].iloc[0]
        parts.append(
            f"Product/Region: top product is {product['Product']} ({_fmt_num(product['total_sales'])}), top region is {region['Region']} ({_fmt_num(region['total_sales'])})."
        )

    if "demographics" in keyword_retriever(question):
        gender = outputs["gender_segmentation"].sort_values("total_sales", ascending=False).iloc[0]
        age = outputs["age_segmentation"].sort_values("total_sales", ascending=False).iloc[0]
        parts.append(
            f"Demographics: top gender segment is {gender['Customer_Gender']} ({_fmt_num(gender['total_sales'])}), top age group is {age['Age_Group']} ({_fmt_num(age['total_sales'])})."
        )

    if "statistics" in keyword_retriever(question):
        kpi = outputs["kpi_overview"]
        kpi_map = dict(zip(kpi["metric"], kpi["value"]))
        parts.append(
            "Statistics: "
            f"mean sales={_fmt_num(kpi_map.get('avg_sales', 0))}, "
            f"median sales={_fmt_num(kpi_map.get('median_sales', 0))}, "
            f"std sales={_fmt_num(kpi_map.get('std_sales', 0))}."
        )

    if not parts:
        return "Could not map this question to a known metric. Try asking about time, product, region, demographics, or statistics."
    return "\n\n".join(parts)


with st.sidebar:
    st.header("Data Setup")
    csv_path = st.text_input("CSV path", value=DEFAULT_CSV)
    st.caption("This app assumes analysis-ready data; minimal cleaning is applied.")


if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

api_key = os.getenv("OPENAI_API_KEY")
invalid_key = (not api_key) or api_key.strip().lower() in {"your_key_here", "sk-your-key-here"}
client = OpenAI(api_key=api_key) if not invalid_key else None

try:
    df, outputs, kb_text = build_kb(csv_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()


summary_tab, kb_tab, rag_tab, visuals_tab = st.tabs(
    [
        "Advanced Summary",
        "Knowledge Base",
        "RAG Chat + Memory",
        "Visuals + Ranking",
    ]
)

with summary_tab:
    st.subheader("1) Sales Performance by Time Period")
    c1, c2, c3 = st.columns(3)
    monthly = outputs["monthly_trend"]
    yearly = outputs["yearly_trend"]
    c1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
    c2.metric("Best Month", str(monthly.sort_values('total_sales', ascending=False).iloc[0]["YearMonth"]))
    c3.metric("Best Year", str(int(yearly.sort_values('total_sales', ascending=False).iloc[0]["Year"])))
    st.line_chart(monthly.set_index("YearMonth")["total_sales"])

    st.subheader("2) Product and Regional Analysis")
    p1, p2 = st.columns(2)
    product = outputs["product_summary"]
    region = outputs["region_summary"]
    p1.dataframe(product[["Product", "total_sales", "avg_sales", "median_sales", "std_sales"]], use_container_width=True)
    p2.dataframe(region[["Region", "total_sales", "avg_sales", "median_sales", "std_sales"]], use_container_width=True)

    st.subheader("3) Customer Segmentation by Demographics")
    d1, d2 = st.columns(2)
    d1.dataframe(outputs["age_segmentation"], use_container_width=True)
    d2.dataframe(outputs["gender_segmentation"], use_container_width=True)

    st.subheader("4) Statistical Measures")
    kpi = outputs["kpi_overview"]
    st.dataframe(kpi, use_container_width=True)

with kb_tab:
    st.subheader("Knowledge Base (Structured for Retrieval)")
    for sid in SECTION_IDS:
        st.markdown(f"**{sid}**")
        st.write(kb_text[sid])

    st.markdown("**Retriever-Ready Tables**")
    st.write("`monthly_trend`, `quarterly_trend`, `yearly_trend`, `product_summary`, `region_summary`, `age_segmentation`, `gender_segmentation`, `kpi_overview`")

with rag_tab:
    st.subheader("RAG Assistant")
    if not client:
        st.warning("Set a valid `OPENAI_API_KEY` (not `your_key_here`) to enable LLM query planning and answer generation.")

    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about trends, products, regions, demographics, or statistics")
    if user_q:
        st.session_state.rag_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        memory_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.rag_messages[-6:]])

        if client:
            try:
                selected_sections = llm_section_selector(client, user_q, memory_text)
            except Exception:
                selected_sections = keyword_retriever(user_q)
        else:
            selected_sections = keyword_retriever(user_q)

        st.info(f"Retrieved sections: {', '.join(selected_sections)}")

        if client:
            try:
                answer = answer_with_rag(
                    client=client,
                    question=user_q,
                    selected_sections=selected_sections,
                    kb_text=kb_text,
                    memory_messages=st.session_state.rag_messages,
                )
            except Exception as exc:
                answer = (
                    f"LLM call failed: {exc}\n\n"
                    "Local fallback answer:\n"
                    f"{local_fallback_answer(user_q, df, outputs)}"
                )
        else:
            answer = "LLM is disabled. Local fallback answer:\n\n" + local_fallback_answer(user_q, df, outputs)

        st.session_state.rag_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

with visuals_tab:
    st.subheader("Saved Visuals")
    vis_files = [
        "analysis_output/visual_1_sales_time_period.png",
        "analysis_output/visual_2_product_region_analysis.png",
        "analysis_output/visual_3_customer_segmentation.png",
        "analysis_output/visual_4_statistical_measures.png",
        "analysis_output/sales_dashboard.png",
    ]

    for vf in vis_files:
        p = Path(vf)
        if p.exists():
            st.image(str(p), caption=vf)

    st.subheader("Product Ranking (Top to Bottom)")
    product_rank = (
        df.groupby("Product", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .reset_index(drop=True)
    )
    product_rank["Rank"] = product_rank.index + 1
    product_rank["Sales_%"] = (product_rank["Sales"] / product_rank["Sales"].sum() * 100).round(2)
    st.dataframe(product_rank[["Rank", "Product", "Sales", "Sales_%"]], use_container_width=True)

    st.markdown("**KPI Cards**")
    for _, row in product_rank.iterrows():
        st.metric(
            label=f"#{int(row['Rank'])} {row['Product']}",
            value=f"{int(row['Sales']):,}",
            delta=f"{row['Sales_%']:.2f}% of product sales",
        )
