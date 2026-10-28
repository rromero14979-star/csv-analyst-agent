# Test scenarios (for the evaluator)

Open the app and paste your OpenAI API key in the sidebar (a Tavily key is only needed for web search). The demo
dataset is already loaded: 2,000 synthetic sales orders. Type each question, or click the matching example button
(S1–S5). After each answer, open **Agent path** to see which agents ran.

| # | Question | Expected behaviour | What to look at |
|---|---|---|---|
| S1 | Which region had the highest total revenue in 2024? | Route **numeric**. The answer is **North** (248,074.33). Code is shown, and every number in the answer comes from the result | Path: schema_profiler → question_router → pandas_agent → code_executor → answer_composer → answer_validator → report_builder |
| S2 | Plot monthly revenue as a line chart. | Route **chart**. A line chart with a title and axis labels, plus the monthly totals | Path contains chart_agent; an image appears |
| S3 | What does the Discount column mean? | Route **concept**. A fraction between 0 and 1; a missing value means "not recorded" | Path contains dictionary_rag; no code is run |
| S4 | What is the average profit margin by product? | There is no cost or margin column. The first code fails, the **repair loop** runs, and the bot answers honestly that it cannot be calculated. If the code keeps failing (3 attempts), it ends in fallback_response and lists the columns | Path shows code_executor more than once |
| S5 | Write me a poem about summer. | Route **out_of_scope**. A polite refusal: the bot only works on the loaded table | Path: schema_profiler → question_router → fallback_response |

Optional checks:

- *What is a typical gross margin for retail furniture businesses?* — route **external**, with web sources cited
  (needs a Tavily key).
- Upload `data/external/superstore.csv` or `data/external/dirty_cafe_sales.csv` and ask, for example, *Which region
  had the highest total sales in 2017?* (West) or *What is the sum of Total Spent, ignoring empty, ERROR and UNKNOWN
  values?* (84,763.5).

**Conditional edges exercised:** C1 (S1–S5), C2 repair (S4), C3 pass (S1–S3). The C3 branches `redraft`, `recode_*`
and `give_up` depend on the model making a mistake, so `tests/test_routes.py` (P11–P14) and
`tests/test_integration.py` cover them deterministically.
