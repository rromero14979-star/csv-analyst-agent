# Test protocol — CSV Analyst Agent

Step-by-step checks to verify the project. Each step has a **pass condition**. If a step fails, fix it before you go
on. The short list of test questions for the evaluator is in `TEST_SCENARIOS.md`.

## Step 0 — Set up (15 min)

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                 # macOS/Linux: cp .env.example .env
# open .env and paste your OPENAI_API_KEY and TAVILY_API_KEY
```

**Pass:** `python -c "import analyst.factory"` shows no error.

## Step 1 — Automatic tests, no cost (2 min)

```bash
pytest -q
```

**Pass:** `113 passed`. Without calling any API, the tests show that:

- the graph has 11 nodes, and every branch of C1, C2 and C3 runs;
- the sandbox blocks imports, `open`, `__` tricks, `eval`, file writes and endless loops;
- the full paths work: repair, rewritten answer, rewritten code and the ablations;
- errors are handled (E1–E9), real-world CSVs load (`;` separator, decimal comma, `$` amounts) and the app works.

## Step 2 — Smoke test with real keys (5 min, a few cents)

```bash
python -m analyst.cli "Which region had the highest total revenue in 2024?"
python -m analyst.cli "Plot monthly revenue as a line chart."
python -m analyst.cli "What does the Discount column mean?"
python -m analyst.cli "What is the average profit margin by product?"
python -m analyst.cli "Write me a poem about summer."
```

| Question | Expected |
|---|---|
| S1 | Route `numeric`; the answer names the region; the code is shown |
| S2 | Route `chart`; in the app, a chart appears |
| S3 | Route `concept`; no code is run |
| S4 | An honest "it cannot be calculated", because there is no cost or margin column |
| S5 | Route `out_of_scope`; a polite refusal |

Check S1 by hand:
`python -c "import pandas as pd; d=pd.read_csv('data/sales_synthetic.csv'); print(d[d['Order Date'].str[:4]=='2024'].groupby('Region')['Revenue'].sum())"`
(expected: North, 248,074.33).

**Pass:** S1–S5 behave as in the table, and S1 matches the hand check.

## Step 3 — The app on your computer (15 min)

```bash
streamlit run app.py
```

| Check | Expected |
|---|---|
| Data preview and data dictionary are visible | Yes |
| Example buttons S1–S5 | Each one asks its question |
| Answer, table, chart, "Code used" and "Agent path" | All shown |
| Empty question, or more than 500 characters | A warning, no crash |
| Upload a small CSV of your own and ask | Works; the dictionary is written automatically |
| Upload an empty file, a file over 5 MB, or a non-CSV file | A clear message, no crash |
| No OpenAI key anywhere | The sidebar asks for your OpenAI key |
| OpenAI key but no Tavily key | Web search is turned off, with a note |
| Web search off, then an outside question | A refusal that says web search is off |

**Pass:** every row behaves as expected.

## Step 4 — Evaluation (about 25 min, under 2 USD)

Run at most 3 evaluations at the same time. Each one starts its own sandbox processes, and with more the computer can
run out of memory.

```bash
# demo data: 30 questions
python eval/build_questions.py
python eval/run_eval.py --configs b0,b1,full,a1,a2,a3

# real CSVs: Superstore (25 questions) and dirty cafe sales (21 questions)
python eval/build_questions_external.py superstore data/external/superstore.csv
python eval/build_questions_external.py cafe data/external/dirty_cafe_sales.csv
python eval/run_eval.py --csv data/external/superstore.csv --questions eval/questions_superstore.csv \
       --results eval/results/superstore --dictionary eval/results/superstore/dictionary.json
python eval/run_eval.py --csv data/external/dirty_cafe_sales.csv --questions eval/questions_cafe.csv \
       --results eval/results/cafe --dictionary eval/results/cafe/dictionary.json

# stress test: 9 damaged copies of Superstore x 5 questions
python eval/stress_test.py data/external/superstore.csv
```

- The correct answers are calculated with pandas; nothing is labelled by hand.
- Each configuration writes one CSV per run (answer, route, attempts, time) and a `summary.md`.
- The data dictionary of an uploaded CSV is written by the model once and saved, so every configuration uses the same one.

**Pass:** `full` beats `b0` and `b1`, and no result row contains `Connection error`, `MemoryError`, `exit code` or
`Error code: 429`. If one does, the run is not valid: run it again.

## Step 5 — Deploy (15 min)

1. Push the project to GitHub. Check that `.env` is **not** in the repository.
2. On share.streamlit.io, create an app from the repository, with `app.py` and Python 3.13.
3. Leave **Secrets** empty: visitors type their own keys in the sidebar.
4. Make the app **Public** and open it in a private browser window.

**Pass:** S1–S5 give the same results online as on your computer, and each answer takes about 5–10 seconds.

## Step 6 — Error analysis

From `eval/results/final/`, pick 3 correct and 3 wrong answers and read their `trace`. Classify each error (wrong route,
wrong column, date filter, missing values, answer blocked by the validator, judge too strict). Compare `full` with
`a1`, `a2` and `a3` to see what the repair loop, the validator and the router add.

## Final checklist

- [ ] The public app opens without logging in and answers S1–S5.
- [ ] The code, `requirements.txt` and `README.md` are on GitHub.
- [ ] `pytest -q` passes in a clean environment.
- [ ] The report uses the real numbers from `eval/results/final/`.
- [ ] There are no API keys in the repository or in its history.

## If something goes wrong

| Problem | Likely cause | What to do |
|---|---|---|
| The deployment does not start | A package version does not exist for that Python | Read the Streamlit log; use Python 3.13 |
| Every code run fails online | The sandbox cannot start its process | Read the "Execution failed" message; try a simple question |
| Many answers blocked by the validator | The number check is strict | Read `validation_feedback`; adjust `COMPOSER_SYSTEM` |
| The router picks the wrong route | The prompt is ambiguous | Add one example per route to `ROUTER_SYSTEM` |
| Web search fails | Tavily quota or network | Turn web search off; the other routes keep working |
