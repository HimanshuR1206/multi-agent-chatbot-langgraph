"""Free web + image search via DuckDuckGo (no API key needed)."""
try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS


def web_search(query: str, n: int = 5) -> list:
    try:
        return list(DDGS().text(query, max_results=n))
    except Exception:
        return []


def image_search(query: str, n: int = 4) -> list:
    try:
        return list(DDGS().images(query, max_results=n))
    except Exception:
        return []
