from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import feedparser
import pandas as pd
import requests


RSS_FEEDS = {
    "news_coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "news_cointelegraph": "https://cointelegraph.com/rss",
    "news_decrypt": "https://decrypt.co/feed",
    "news_bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "news_theblock": "https://www.theblock.co/rss.xml",
}


@dataclass(frozen=True)
class TextItem:
    published_at: str
    source: str
    text: str
    url: str | None = None


def fetch_rss_items(feeds: dict[str, str] | Iterable[str] = RSS_FEEDS, timeout: int = 20) -> list[TextItem]:
    rows: list[TextItem] = []
    items = feeds.items() if isinstance(feeds, dict) else [(f"rss_{i}", feed) for i, feed in enumerate(feeds)]
    for source_name, feed in items:
        parsed = feedparser.parse(feed)
        for item in parsed.entries[:50]:
            title = getattr(item, "title", "") or ""
            summary = getattr(item, "summary", "") or ""
            text = f"{title} {summary}".strip()
            if "bitcoin" not in text.lower() and "btc" not in text.lower() and "crypto" not in text.lower():
                continue
            published = getattr(item, "published", None) or getattr(item, "updated", None) or datetime.now(timezone.utc).isoformat()
            rows.append(TextItem(published_at=str(published), source=source_name, text=text[:2000], url=getattr(item, "link", None)))
    return rows


def fetch_gdelt_items(timeout: int = 20) -> list[TextItem]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "bitcoin OR BTC",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 75,
        "sort": "hybridrel",
    }
    try:
        data = requests.get(url, params=params, timeout=timeout).json()
    except Exception:
        return []
    rows = []
    for art in data.get("articles", []):
        text = f"{art.get('title', '')} {art.get('seendate', '')}".strip()
        if text:
            rows.append(TextItem(published_at=art.get("seendate", datetime.now(timezone.utc).isoformat()), source="gdelt", text=text[:2000], url=art.get("url")))
    return rows


def collect_live_text() -> pd.DataFrame:
    items = fetch_rss_items() + fetch_gdelt_items()
    if not items:
        return pd.DataFrame(columns=["published_at", "source", "text", "url"])
    df = pd.DataFrame([item.__dict__ for item in items]).drop_duplicates(subset=["source", "text"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at", "text"])
    return df.sort_values("published_at")
