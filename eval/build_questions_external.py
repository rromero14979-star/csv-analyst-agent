"""Questions for datasets that are not part of the repository (generalization and robustness experiments).

Ground truth is COMPUTED with pandas from the CSV, as in eval/build_questions.py.

    python eval/build_questions_external.py superstore "path/to/Sample - Superstore.csv"
    python eval/build_questions_external.py cafe path/to/dirty_cafe_sales.csv

Datasets:
    superstore  Kaggle "Superstore" sample (9,994 order lines, 21 columns, clean but wide)
    cafe        Kaggle "Cafe Sales - Dirty Data for Cleaning Training" (10,000 rows; numbers stored as text,
                with 'ERROR' / 'UNKNOWN' placeholders and empty cells). Questions say how to treat invalid values.
Writes eval/questions_<dataset>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.profiling import load_dataframe  # noqa: E402


class _Rows(list):
    def add(self, group, question, route, etype, value="", text="", columns="", keywords=""):
        self.append({
            "id": len(self) + 1, "group": group, "question": question, "expected_route": route,
            "expected_type": etype, "expected_value": value, "expected_text": text,
            "expected_columns": columns, "expected_keywords": keywords,
        })


def superstore(df: pd.DataFrame) -> pd.DataFrame:
    y = df["Order Date"].dt.year
    q = _Rows()
    # ---- numeric (11)
    q.add("numeric", "What is the total sales?", "numeric", "number", value=round(df["Sales"].sum(), 2))
    q.add("numeric", "Which region had the highest total sales in 2017?", "numeric", "text",
          text=df[y == 2017].groupby("Region")["Sales"].sum().idxmax())
    q.add("numeric", "What is the total profit of the Furniture category?", "numeric", "number",
          value=round(df[df["Category"] == "Furniture"]["Profit"].sum(), 2))
    q.add("numeric", "Which sub-category has the lowest total profit?", "numeric", "text",
          text=df.groupby("Sub-Category")["Profit"].sum().idxmin())
    q.add("numeric", "What is the average discount in the Central region?", "numeric", "number",
          value=round(df[df["Region"] == "Central"]["Discount"].mean(), 4))
    q.add("numeric", "Which state has the highest total profit?", "numeric", "text",
          text=df.groupby("State")["Profit"].sum().idxmax())
    q.add("numeric", "How many order lines were shipped with the Same Day ship mode?", "numeric", "number",
          value=int((df["Ship Mode"] == "Same Day").sum()))
    q.add("numeric", "Which customer generated the most sales?", "numeric", "text",
          text=df.groupby("Customer Name")["Sales"].sum().idxmax())
    q.add("numeric", "Which month of 2017 had the highest total sales?", "numeric", "text",
          text=df[y == 2017].groupby(df["Order Date"].dt.month_name())["Sales"].sum().idxmax())
    q.add("numeric", "What are the total sales of the Phones sub-category in the East region?", "numeric", "number",
          value=round(df[(df["Sub-Category"] == "Phones") & (df["Region"] == "East")]["Sales"].sum(), 2))
    q.add("numeric", "What is the total quantity sold in the Technology category?", "numeric", "number",
          value=int(df[df["Category"] == "Technology"]["Quantity"].sum()))
    # ---- tricky: rows vs orders, negative values, ratios, dates (5)
    q.add("tricky", "How many unique orders are there?", "numeric", "number", value=int(df["Order ID"].nunique()))
    q.add("tricky", "What percentage of order lines have a negative profit?", "numeric", "number",
          value=round((df["Profit"] < 0).mean() * 100, 2))
    q.add("tricky", "What is the profit margin (total profit divided by total sales) of the Technology category?",
          "numeric", "number",
          value=round(df[df["Category"] == "Technology"]["Profit"].sum() / df[df["Category"] == "Technology"]["Sales"].sum(), 4))
    q.add("tricky", "What is the average number of days between the order date and the ship date?", "numeric", "number",
          value=round((df["Ship Date"] - df["Order Date"]).dt.days.mean(), 2))
    q.add("tricky", "How much did total sales grow from 2016 to 2017?", "numeric", "number",
          value=round(df[y == 2017]["Sales"].sum() - df[y == 2016]["Sales"].sum(), 2))
    # ---- chart (3)
    q.add("chart", "Show a bar chart of total sales by category.", "chart", "chart", columns="Category|Sales")
    q.add("chart", "Plot monthly profit as a line chart.", "chart", "chart", columns="Order Date|Profit")
    q.add("chart", "Plot a histogram of the discount.", "chart", "chart", columns="Discount")
    # ---- concept (2) - the dictionary is drafted by the LLM, there is no curated one
    q.add("concept", "What does the Segment column mean?", "concept", "keywords", keywords="customer|consumer|corporate")
    q.add("concept", "What does a negative value in the Profit column mean?", "concept", "keywords", keywords="loss|lost|negative")
    # ---- external (1)
    q.add("external", "What is a typical net profit margin for office supply retailers?", "external", "web")
    # ---- decline (3)
    q.add("decline", "Write me a poem about shipping.", "out_of_scope", "decline")
    q.add("decline", "Delete all orders from Texas.", "out_of_scope", "decline")
    q.add("decline", "What is the average customer age by segment?", "numeric", "decline")
    return pd.DataFrame(q)


def cafe(df: pd.DataFrame) -> pd.DataFrame:
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in ("Quantity", "Price Per Unit", "Total Spent")}
    bad = {"ERROR", "UNKNOWN"}

    def valid(col):
        return df[col].where(~df[col].isin(bad))

    item, pay, loc = valid("Item"), valid("Payment Method"), valid("Location")
    dates = pd.to_datetime(df["Transaction Date"], errors="coerce")
    clause = "ignoring empty, ERROR and UNKNOWN values"
    q = _Rows()
    # ---- numeric (8)
    q.add("numeric", "How many transactions are there?", "numeric", "number", value=len(df))
    q.add("numeric", f"What is the sum of Total Spent, {clause}?", "numeric", "number", value=round(num["Total Spent"].sum(), 2))
    q.add("numeric", f"Which item appears in the most transactions, {clause}?", "numeric", "text", text=item.value_counts().idxmax())
    q.add("numeric", f"What is the average quantity per transaction, {clause}?", "numeric", "number", value=round(num["Quantity"].mean(), 4))
    q.add("numeric", f"Which payment method is used most often, {clause}?", "numeric", "text", text=pay.value_counts().idxmax())
    q.add("numeric", f"What is the total quantity of Coffee sold, {clause}?", "numeric", "number",
          value=int(num["Quantity"][item == "Coffee"].sum()))
    q.add("numeric", f"What is the average Total Spent of In-store transactions, {clause}?", "numeric", "number",
          value=round(num["Total Spent"][loc == "In-store"].mean(), 4))
    q.add("numeric", f"Which month had the highest sum of Total Spent, {clause}?", "numeric", "text",
          text=num["Total Spent"].groupby(dates.dt.month_name()).sum().idxmax())
    # ---- tricky: data quality questions (4)
    q.add("tricky", "How many transactions have an invalid Total Spent value (empty, ERROR or UNKNOWN)?", "numeric", "number",
          value=int(num["Total Spent"].isna().sum()))
    q.add("tricky", "What percentage of transactions have a missing or invalid Location (empty, ERROR or UNKNOWN)?",
          "numeric", "number", value=round(loc.isna().mean() * 100, 2))
    q.add("tricky", f"Recompute Total Spent as Quantity times Price Per Unit, {clause}. What is the sum of the recomputed values?",
          "numeric", "number", value=round((num["Quantity"] * num["Price Per Unit"]).sum(), 2))
    q.add("tricky", f"What is the most common price per unit of a Smoothie, {clause}?", "numeric", "number",
          value=float(num["Price Per Unit"][item == "Smoothie"].mode().iloc[0]))
    # ---- chart (3)
    q.add("chart", "Show a bar chart of the number of transactions per item.", "chart", "chart", columns="Item")
    q.add("chart", f"Plot the monthly sum of Total Spent as a line chart, {clause}.", "chart", "chart",
          columns="Transaction Date|Total Spent")
    q.add("chart", "Show a bar chart of the number of transactions by payment method.", "chart", "chart", columns="Payment Method")
    # ---- concept (2)
    q.add("concept", "What does the Location column mean?", "concept", "keywords", keywords="in-store|takeaway|where")
    q.add("concept", "What does the Price Per Unit column mean?", "concept", "keywords", keywords="price|unit|item")
    # ---- external (1)
    q.add("external", "What is the average price of a cup of coffee in the United States?", "external", "web")
    # ---- decline (3)
    q.add("decline", "Write me a poem about coffee.", "out_of_scope", "decline")
    q.add("decline", "Replace all ERROR values with 0 in the file.", "out_of_scope", "decline")
    q.add("decline", "What is the average age of the customers?", "numeric", "decline")
    return pd.DataFrame(q)


BUILDERS = {"superstore": superstore, "cafe": cafe}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    parser.add_argument("csv")
    args = parser.parse_args()
    out = BUILDERS[args.dataset](load_dataframe(args.csv))
    path = ROOT / "eval" / f"questions_{args.dataset}.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}: {out['group'].value_counts().to_dict()}")
    print(out[["id", "question", "expected_value", "expected_text"]].to_string(index=False))
