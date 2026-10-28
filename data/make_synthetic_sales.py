"""Generate the synthetic sales dataset used by the demo and the experiments.

Deterministic (fixed seed) so ground truth is reproducible:  python data/make_synthetic_sales.py
Column names intentionally contain spaces and the table has missing values and real dates,
which makes the code-repair loop of the agent meaningful.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000

PRODUCTS = {  # product: (category, base price)
    "Laptop": ("Electronics", 950.0), "Phone": ("Electronics", 620.0), "Headphones": ("Electronics", 120.0),
    "Desk": ("Furniture", 340.0), "Chair": ("Furniture", 180.0), "Bookshelf": ("Furniture", 210.0),
    "Notebook": ("Office Supplies", 6.0), "Pen Set": ("Office Supplies", 14.0), "Stapler": ("Office Supplies", 22.0),
}
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.22, 0.18, 0.2, 0.25, 0.15]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
DISCOUNTS = [0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def make(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    products = rng.choice(list(PRODUCTS), size=n)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.randint(0, 731, size=n), unit="D")
    unit_price = np.array([PRODUCTS[p][1] for p in products]) * rng.uniform(0.95, 1.05, size=n)
    units = rng.poisson(2.5, size=n) + 1
    discount = rng.choice(DISCOUNTS, size=n).astype(float)
    revenue = units * unit_price * (1 - discount)

    df = pd.DataFrame({
        "Order ID": [f"ORD-{i:05d}" for i in range(1, n + 1)],
        "Order Date": dates.strftime("%Y-%m-%d"),
        "Region": rng.choice(REGIONS, size=n, p=REGION_WEIGHTS),
        "Category": [PRODUCTS[p][0] for p in products],
        "Product": products,
        "Units": units,
        "Unit Price": unit_price.round(2),
        "Discount": discount,
        "Revenue": revenue.round(2),
        "Customer Segment": rng.choice(SEGMENTS, size=n, p=[0.5, 0.3, 0.2]),
    }).sort_values("Order Date").reset_index(drop=True)
    df["Order ID"] = [f"ORD-{i:05d}" for i in range(1, n + 1)]
    # missing values (3% discounts not recorded, 2% customers not classified)
    df.loc[rng.rand(n) < 0.03, "Discount"] = np.nan
    df.loc[rng.rand(n) < 0.02, "Customer Segment"] = np.nan
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sales_synthetic.csv"
    make().to_csv(out, index=False)
    print(f"Wrote {out}")
