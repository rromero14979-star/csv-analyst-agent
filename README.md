# CSV Analyst Agent

A chatbot that answers questions about a CSV table. It does not guess the numbers: it writes Python code, runs the
code on your data, fixes the code if it breaks, and checks every number before it shows the answer.

Built with LangChain, LangGraph and Streamlit.

**Live demo:** https://csv-analyst-agt.streamlit.app/ (paste your own OpenAI API key in the sidebar)

## What you can ask

| Type of question | Example | What the bot does |
|---|---|---|
| A number from the table | "Which region had the highest total revenue in 2024?" | Writes and runs pandas code |
| A chart | "Plot monthly revenue as a line chart." | Writes and runs matplotlib code |
| What a column means | "What does the Discount column mean?" | Looks it up in the data dictionary |
| Outside information | "What is a typical gross margin for furniture shops?" | Searches the web and cites sources |
| Something it should not do | "Write me a poem." / "Delete all rows." | Says no, politely |

If the question needs a column that does not exist, the bot says so and lists the columns it has.

## How to use the demo

1. Open the demo link.
2. In the sidebar, paste your **OpenAI API key**. A **Tavily key** is optional: without it, web search is off.
   Your keys stay in your browser session and are never saved.
3. Use the demo sales data (already loaded), or upload your own CSV (up to 5 MB).
4. Type a question, or click one of the example buttons S1–S5.
5. Under each answer, open **Code used** and **Agent path** to see how the answer was made.

## How it works

The bot is a graph of 11 small agents (nodes). Three decision points (conditional edges) choose the next step.

```
question -> schema_profiler -> question_router
                                  |-- numeric  -> pandas_agent -+
                                  |-- chart    -> chart_agent  -+-> code_executor -> answer_composer -> answer_validator -> report_builder
                                  |-- concept  -> dictionary_rag ---------------------^                        |
                                  |-- external -> web_search -------------------------^                        |
                                  |-- out_of_scope -> fallback_response <---------------- (cannot fix it) -------+
```

| # | Node | What it does |
|---|---|---|
| 1 | `schema_profiler` | Reads the column names, types and simple statistics. It never sends raw rows to the model. |
| 2 | `question_router` | Decides the type of question. |
| 3 | `pandas_agent` | Writes pandas code to calculate the answer. |
| 4 | `chart_agent` | Writes matplotlib code to draw the chart. |
| 5 | `code_executor` | Runs the code in a safe sandbox. |
| 6 | `dictionary_rag` | Finds what a column means in the data dictionary. |
| 7 | `web_search` | Finds outside information with Tavily. |
| 8 | `answer_composer` | Writes the answer, using only the result or the sources. |
| 9 | `answer_validator` | Checks the answer: every number must come from the result, then an LLM judge reviews it. |
| 10 | `report_builder` | Builds the final answer with the code and the agent path. |
| 11 | `fallback_response` | Gives an honest "I can't" when the request cannot be done safely. |

The three decisions:

| Edge | Question it answers | Possible next steps |
|---|---|---|
| C1, after `question_router` | What kind of question is it? | pandas, chart, dictionary, web, or refuse |
| C2, after `code_executor` | Did the code run? | answer it · try again (up to 3 times) · give up |
| C3, after `answer_validator` | Is the answer supported? | show it · rewrite the answer · rewrite the code (once) · give up |

Every loop has a limit, so the bot always stops.

**Shared state.** All nodes read and write one dictionary (`analyst/state.py`): the question, the profile, the code,
the result, the answer, the validation result and the path taken.

**Sandbox.** The generated code runs in a separate Python process. It cannot import modules, open files, read the API
keys, or run for more than 15 seconds.

## Data

- **Demo data** (`data/sales_synthetic.csv`): 2,000 synthetic sales orders, made by `data/make_synthetic_sales.py`. It
  has column names with spaces, some missing values and real dates, to make the task realistic.
- **Data dictionary** (`data/sales_dictionary.json`): describes each demo column. For an uploaded CSV, the model
  writes the dictionary automatically.
- **Real CSVs for testing** (`data/external/`): the Kaggle "Superstore" sample and a deliberately dirty Kaggle cafe
  sales file. Sources are in `data/external/SOURCES.md`.
- **Your CSV**: must have a header row, at most 50,000 rows, 60 columns and 5 MB. The loader also handles `;` as the
  separator, decimal commas, `$1,234.56` amounts and title lines above the header.

## Run it on your computer

You need Python 3.13 (the version used for all tests; 3.12 should also work).

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                 # macOS/Linux: cp .env.example .env
# open .env and paste your OPENAI_API_KEY and TAVILY_API_KEY

streamlit run app.py                   # the chat app
python -m analyst.cli "Which region had the highest total revenue in 2024?"   # one question, in the terminal
```

## Tests

```bash
pytest -q
```

113 tests run without calling any API, so they cost nothing. They check every node, all 14 branches of the three
edges, the sandbox rules, error handling, CSV loading and the Streamlit app.

## Results

We asked 76 questions on three datasets. The correct answers were calculated with pandas, not typed by hand.
Model: `gpt-4.1-mini`, 29 Oct 2026.

| Dataset (questions) | b0 | b1 | **full** | a1 | a2 | a3 |
|---|---|---|---|---|---|---|
| Synthetic demo (30) | 26.7% | 56.7% | **100%** | 100% | 96.7% | 56.7% |
| Kaggle Superstore (25) | 24.0% | 76.0% | **100%** | 100% | 100% | 76.0% |
| Kaggle dirty cafe sales (21) | 38.1% | 66.7% | **100%** | 100% | 100% | 76.2% |

- **b0**: the model alone, with no code. **b1**: code written once, with no checks.
- **full**: the complete bot. **a1 / a2 / a3**: the bot without the repair loop / the validator / the router.

What we learned:

- The full bot answered all 76 questions correctly and never showed an invented number.
- The **router** matters most: without it, accuracy falls by 24 to 43 points.
- The **repair loop** and the **validator** are safety nets. On these questions they changed at most one answer.
  Without the validator, the bot once invented a profit margin.
- A **stress test** with 9 damaged copies of a real CSV went from 27/45 to 45/45 correct answers after we improved
  the CSV loader.

How to repeat the experiments (keep 3 runs at most at the same time, or the computer may run out of memory):

```bash
python eval/build_questions.py                      # questions for the demo data
python eval/run_eval.py --configs b0,b1,full,a1,a2,a3
python eval/build_questions_external.py superstore data/external/superstore.csv
python eval/run_eval.py --csv data/external/superstore.csv --questions eval/questions_superstore.csv \
       --results eval/results/superstore --dictionary eval/results/superstore/dictionary.json
python eval/stress_test.py data/external/superstore.csv
```

Detailed results are in `eval/results/final/` and `eval/results/stress/`.

## Limits

- The validator checks that the answer matches the result. It cannot tell whether the result itself makes sense:
  if the code invents a metric, the answer can still pass.
- The router sometimes picks the wrong type of question.
- Each question is answered on its own: the bot has no memory of earlier questions.
- The sandbox lowers the risk but is not a full security boundary. A production system should use a container.
- One table at a time; no joins between files.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository. `.env` is ignored, so your keys are not uploaded.
2. Go to share.streamlit.io → **New app** → choose the repository and `app.py`.
3. In **Advanced settings**, choose Python 3.13. Leave **Secrets** empty: visitors use their own keys.
4. In **Settings → Sharing**, make the app **Public**.

More documents: `TEST_SCENARIOS.md` (5 test questions for the evaluator) and `TEST_PROTOCOL.md` (step-by-step checks).
