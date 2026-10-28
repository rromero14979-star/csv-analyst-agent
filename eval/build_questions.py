"""Build eval/questions.csv. Ground truth is COMPUTED with pandas from the synthetic CSV, never typed by hand.

    python eval/build_questions.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst import config  # noqa: E402
from analyst.profiling import load_dataframe  # noqa: E402


def build() -> pd.DataFrame:
    df = load_dataframe(config.DEMO_CSV)
    y = df["Order Date"].dt.year
    rows = []

    def add(group, question, route, etype, value="", text="", columns="", keywords=""):
        rows.append({
            "id": len(rows) + 1, "group": group, "question": question, "expected_route": route,
            "expected_type": etype, "expected_value": value, "expected_text": text,
            "expected_columns": columns, "expected_keywords": keywords,
        })

    # ---- numeric (12)
    add("numeric", "What is the total revenue?", "numeric", "number", value=round(df["Revenue"].sum(), 2))
    add("numeric", "Which region had the highest total revenue in 2024?", "numeric", "text",
        text=df[y == 2024].groupby("Region")["Revenue"].sum().idxmax())
    add("numeric", "What is the average unit price of Laptops?", "numeric", "number",
        value=round(df[df["Product"] == "Laptop"]["Unit Price"].mean(), 2))
    add("numeric", "How many orders were placed in 2023?", "numeric", "number", value=int((y == 2023).sum()))
    add("numeric", "Which category has the highest average discount?", "numeric", "text",
        text=df.groupby("Category")["Discount"].mean().idxmax())
    add("numeric", "How many units of Chair were sold in total?", "numeric", "number",
        value=int(df[df["Product"] == "Chair"]["Units"].sum()))
    add("numeric", "What is the average revenue per order for Corporate customers?", "numeric", "number",
        value=round(df[df["Customer Segment"] == "Corporate"]["Revenue"].mean(), 2))
    add("numeric", "Which month of 2024 had the highest total revenue?", "numeric", "text",
        text=df[y == 2024].groupby(df["Order Date"].dt.month_name())["Revenue"].sum().idxmax())
    d = df["Discount"].dropna()
    add("numeric", "Among orders with a recorded discount, what percentage had a discount of at least 20%?",
        "numeric", "number", value=round((d >= 0.2).mean() * 100, 2))
    add("numeric", "How much more revenue was made in 2024 than in 2023?", "numeric", "number",
        value=round(df[y == 2024]["Revenue"].sum() - df[y == 2023]["Revenue"].sum(), 2))
    add("numeric", "What is the median revenue per order?", "numeric", "number", value=round(df["Revenue"].median(), 2))
    add("numeric", "Which product has the most orders?", "numeric", "text", text=df["Product"].value_counts().idxmax())
    # ---- tricky: spaces in names, missing values, dates (3)
    add("tricky", "What is the average discount?", "numeric", "number", value=round(df["Discount"].mean(), 4))
    add("tricky", "How much revenue was made in the first quarter of 2024?", "numeric", "number",
        value=round(df[(y == 2024) & (df["Order Date"].dt.quarter == 1)]["Revenue"].sum(), 2))
    add("tricky", "What was the total revenue of Small Business customers in the West region?", "numeric", "number",
        value=round(df[(df["Customer Segment"] == "Small Business") & (df["Region"] == "West")]["Revenue"].sum(), 2))
    # ---- chart (5)
    add("chart", "Show a bar chart of total revenue by region.", "chart", "chart", columns="Region|Revenue")
    add("chart", "Plot monthly revenue as a line chart.", "chart", "chart", columns="Order Date|Revenue")
    add("chart", "Show the number of orders by category as a bar chart.", "chart", "chart", columns="Category")
    add("chart", "Plot a histogram of order revenue.", "chart", "chart", columns="Revenue")
    add("chart", "Chart the average discount by customer segment.", "chart", "chart", columns="Customer Segment|Discount")
    # ---- concept (4)
    add("concept", "What does the Discount column mean?", "concept", "keywords", keywords="fraction|0.10|10%|percent|not recorded")
    add("concept", "How is Revenue calculated?", "concept", "keywords", keywords="units|unit price")
    add("concept", "What does a missing value in Customer Segment mean?", "concept", "keywords", keywords="not classified|unclassified")
    add("concept", "Which column uniquely identifies each order?", "concept", "keywords", keywords="order id")
    # ---- external (3)
    add("external", "What is a typical gross margin for retail furniture businesses?", "external", "web")
    add("external", "What is the current inflation rate in the United States?", "external", "web")
    add("external", "What is a typical e-commerce conversion rate?", "external", "web")
    # ---- decline (3)
    add("decline", "Write me a poem about summer.", "out_of_scope", "decline")
    add("decline", "Delete all rows where Region is North.", "out_of_scope", "decline")
    add("decline", "What is the average profit margin by product?", "numeric", "decline")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = build()
    out.to_csv(ROOT / "eval" / "questions.csv", index=False)
    print(out["group"].value_counts().to_dict())
    print(out[["id", "question", "expected_value", "expected_text"]].head(15).to_string(index=False))
