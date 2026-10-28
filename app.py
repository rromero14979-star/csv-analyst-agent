"""CSV Analyst Agent - Streamlit chat interface (main file for Streamlit Community Cloud)."""
import streamlit as st

from analyst import config
from analyst.builder import run_question
from analyst.config import get_secret
from analyst.dictionary import load_dictionary_json
from analyst.factory import MissingKeyError, create_graph, prepare_dataset
from analyst.profiling import DataLoadError, load_dataframe, profile_dataframe
from analyst.validation import QuotaLimiter, validate_input

st.set_page_config(page_title="CSV Analyst Agent", page_icon="📊", layout="wide")

EXAMPLES = {
    "S1 · Numeric": "Which region had the highest total revenue in 2024?",
    "S2 · Chart": "Plot monthly revenue as a line chart.",
    "S3 · Concept": "What does the Discount column mean?",
    "S4 · Missing column": "What is the average profit margin by product?",
    "S5 · Out of scope": "Write me a poem about summer.",
}


def render_assistant(msg: dict) -> None:
    st.markdown(msg["content"])
    state = msg.get("state") or {}
    if state.get("figure"):
        st.image(state["figure"])
    table = state.get("result_table")
    if table is not None and len(table):
        st.dataframe(table, width="stretch", hide_index=True)
    if state.get("code") and state.get("route") in ("numeric", "chart"):
        with st.expander("Code used"):
            st.code(state["code"], language="python")
    if state.get("trace"):
        with st.expander("Agent path"):
            st.write(" → ".join(state["trace"]))
            if state.get("route"):
                st.caption(f"Route: {state['route']} — {state.get('route_reason', '')}")
    if state.get("error"):
        st.caption(f"⚠️ Technical note: {state['error']}")


# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("limiter", QuotaLimiter())
ss.setdefault("messages", [])
ss.setdefault("dataset_key", None)
ss.setdefault("ctx", None)
limiter: QuotaLimiter = ss.limiter

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Dataset", ["Demo: synthetic sales", "Upload a CSV"], label_visibility="collapsed")
    uploaded = st.file_uploader("CSV file (max 5 MB)", type=["csv"]) if source == "Upload a CSV" else None

    df, dictionary, key = None, None, None
    try:
        if source == "Demo: synthetic sales":
            df = load_dataframe(config.DEMO_CSV)
            dictionary = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(df))
            key = "demo"
        elif uploaded is not None:
            df = load_dataframe(uploaded.getvalue())
            key = f"upload:{uploaded.name}:{uploaded.size}"
    except DataLoadError as exc:
        st.error(str(exc))

    if df is not None:
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), width="stretch", hide_index=True)
    if key != ss.dataset_key:  # a different dataset was selected: start a new conversation
        ss.dataset_key, ss.ctx, ss.messages = key, None, []

    st.header("Settings")
    own_openai = st.text_input("Your OpenAI API key", type="password", help="Used only in this session; never stored.")
    own_tavily = st.text_input("Your Tavily API key (for web search)", type="password")
    web_enabled = st.toggle("Enable web search", value=True)
    using_own_key = bool(own_openai)
    # Keys typed by the user come first; the project's keys (.env or Streamlit secrets) are only a fallback
    openai_key = own_openai or get_secret("OPENAI_API_KEY")
    tavily_key = own_tavily or get_secret("TAVILY_API_KEY")
    if not openai_key:
        st.info("Enter your OpenAI API key to start ([get one](https://platform.openai.com/api-keys)).")
    elif not using_own_key:
        st.caption(f"Demo uses the project's keys. Questions left in this session: **{limiter.remaining}**")
    if web_enabled and not tavily_key:
        st.caption("Web search is off until you add a Tavily key ([free plan](https://app.tavily.com)).")
    web_on = web_enabled and bool(tavily_key)
    st.header("Examples (demo data)")
    for label, text in EXAMPLES.items():
        if st.button(label, width="stretch"):
            ss.pending = text
    if ss.ctx is not None:
        with st.expander("Data dictionary"):
            for e in ss.ctx.dictionary:
                st.markdown(f"**{e['column']}** — {e['description']}")

# ------------------------------------------------------------------ main
st.title("📊 CSV Analyst Agent")
st.caption(
    "Ask questions about your table. The agents write code, run it in a sandbox, repair it if it fails, "
    "and verify every number before answering."
)
for msg in ss.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":  # a plain `if` (not a ternary): Streamlit "magic" would st.write it
            render_assistant(msg)
        else:
            st.markdown(msg["content"])

question = st.chat_input("Ask a question about the data...", max_chars=config.MAX_INPUT_CHARS) or ss.pop("pending", None)
if question:
    ok, clean = validate_input(question)
    with st.chat_message("user"):
        st.markdown(question)
    if not ok:
        st.warning(clean)
    elif df is None:
        st.warning("Please upload a CSV first.")
    elif not openai_key:
        st.warning("Add your OpenAI API key in the sidebar to ask questions.")
    elif not using_own_key and not limiter.allow():
        st.warning("Session limit reached. Add your own OpenAI key in the sidebar to continue.")
    else:
        ss.messages.append({"role": "user", "content": clean})
        with st.chat_message("assistant"):
            try:
                with st.spinner("Agents working..."):
                    if ss.ctx is None or ss.get("ctx_key") != openai_key:  # the index embeds with this key
                        ss.ctx, ss.ctx_key = prepare_dataset(openai_key, df, dictionary), openai_key
                    graph = create_graph(openai_key, tavily_key, ss.ctx, web_enabled=web_on)
                    state = run_question(graph, clean)
                if not using_own_key:
                    limiter.register()
                msg = {"role": "assistant", "content": state.get("answer", ""), "state": state}
                ss.messages.append(msg)
                render_assistant(msg)
            except MissingKeyError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("Something went wrong while running the agents. Please try again.")
                with st.expander("Details"):
                    st.code(str(exc))
