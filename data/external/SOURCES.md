# External datasets (used only for the generalization and robustness experiments)

| File | Source | Notes |
|---|---|---|
| `superstore.csv` | Kaggle "Superstore Dataset" (Tableau's *Sample - Superstore*) | 9,994 order lines, 21 columns, 2014-2017. Unmodified copy. |
| `dirty_cafe_sales.csv` | Kaggle "Cafe Sales - Dirty Data for Cleaning Training" by ahmedmohamed2003 (copy downloaded from the OVHcloud AI Endpoints tutorial) | 10,000 rows, 8 columns; numbers stored as text with `ERROR` / `UNKNOWN` placeholders and empty cells, on purpose. Unmodified copy. |

Both files belong to their authors and are included only so the experiments in `eval/` can be reproduced.
Check each dataset's Kaggle page for its license before reusing it elsewhere.
