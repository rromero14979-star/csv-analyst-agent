"""Command line runner:  python -m analyst.cli "question" [--csv file.csv] [--no-web]"""
import argparse
import json
import sys

from analyst import config
from analyst.builder import run_question
from analyst.config import get_secret
from analyst.dictionary import load_dictionary_json
from analyst.factory import MissingKeyError, create_graph, prepare_dataset
from analyst.profiling import DataLoadError, load_dataframe, profile_dataframe
from analyst.validation import validate_input


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which cannot print "→"
    parser = argparse.ArgumentParser(description="Ask a question about a CSV")
    parser.add_argument("question")
    parser.add_argument("--csv", default=str(config.DEMO_CSV))
    parser.add_argument("--no-web", action="store_true")
    args = parser.parse_args()

    ok, clean = validate_input(args.question)
    if not ok:
        raise SystemExit(clean)
    try:
        df = load_dataframe(args.csv)
        dictionary = None
        if args.csv == str(config.DEMO_CSV):
            dictionary = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(df))
        ctx = prepare_dataset(get_secret("OPENAI_API_KEY"), df, dictionary)
        graph = create_graph(get_secret("OPENAI_API_KEY"), get_secret("TAVILY_API_KEY"), ctx, web_enabled=not args.no_web)
    except (DataLoadError, MissingKeyError) as exc:
        raise SystemExit(str(exc))
    state = run_question(graph, clean)
    print(state.get("final_report", state.get("answer", "")))
    if state.get("result_table") is not None:
        print("\n", state["result_table"].head(15).to_string(index=False))
    print("\nTRACE:", " -> ".join(state.get("trace", [])))


if __name__ == "__main__":
    main()
