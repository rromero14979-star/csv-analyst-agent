"""Web search through Tavily, with one retry and trimmed content."""
from tavily import TavilyClient

from analyst import config


def make_web_search_fn(api_key: str, max_results: int = config.WEB_MAX_RESULTS):
    client = TavilyClient(api_key=api_key)

    def web_search(query: str) -> list:
        last_error = None
        for _ in range(2):  # one retry
            try:
                res = client.search(query=query, max_results=max_results, search_depth="basic", timeout=15)
                return [
                    {
                        "text": (r.get("content") or "")[: config.WEB_SNIPPET_CHARS],
                        "source": r.get("title") or r.get("url", "web"),
                        "locator": r.get("url", ""),
                        "kind": "web",
                    }
                    for r in res.get("results", [])
                ]
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Web search failed: {last_error}")

    return web_search
