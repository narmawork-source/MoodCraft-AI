import pandas as pd
import streamlit as st

df = pd.read_csv("/Users/narmasarav/Downloads/sales_data.csv")

product_rank = (
    df.groupby("Product", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
    .reset_index(drop=True)
)
product_rank["Rank"] = product_rank.index + 1
product_rank["Sales_%"] = (product_rank["Sales"] / product_rank["Sales"].sum() * 100).round(2)

st.subheader("Product Ranking (Top to Bottom)")
st.dataframe(product_rank[["Rank", "Product", "Sales", "Sales_%"]], use_container_width=True)
for _, row in product_rank.iterrows():
    st.metric(
        label=f"#{row['Rank']} {row['Product']}",
        value=f"{int(row['Sales']):,}",
        delta=f"{row['Sales_%']}% of product sales"
    )
