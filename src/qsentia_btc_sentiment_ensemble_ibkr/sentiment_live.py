from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import re
from typing import Iterable
from urllib.parse import urlencode

import feedparser
import pandas as pd
import requests


RSS_FEEDS = {
    "news_coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "news_cointelegraph": "https://cointelegraph.com/rss",
    "news_cointelegraph_bitcoin": "https://cointelegraph.com/rss/tag/bitcoin",
    "news_decrypt": "https://decrypt.co/feed",
    "news_bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "news_theblock": "https://www.theblock.co/rss.xml",
    "news_cryptoslate": "https://cryptoslate.com/feed/",
    "news_bitcoincom": "https://news.bitcoin.com/feed/",
    "news_yahoo_btc": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
}

SOURCE_ALIASES = {
    "news_cointelegraph_bitcoin": "news_cointelegraph",
    "news_cryptoslate": "gdelt",
    "news_bitcoincom": "gdelt",
    "news_yahoo_btc": "gdelt",
    "news_google_btc": "gdelt",
}

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 QSentiaBot/1.0"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
}


@dataclass(frozen=True)
class TextItem:
    published_at: str
    source: str
    text: str
    url: str | None = None


def _google_news_url(query: str) -> str:
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    return "https://news.google.com/rss/search?" + urlencode(params)


GOOGLE_NEWS_FALLBACK_FEEDS = {
    "news_coindesk": _google_news_url("site:coindesk.com bitcoin OR BTC when:2d"),
    "news_cointelegraph": _google_news_url("site:cointelegraph.com bitcoin OR BTC when:2d"),
    "news_decrypt": _google_news_url("site:decrypt.co bitcoin OR BTC when:2d"),
    "news_bitcoinmagazine": _google_news_url("site:bitcoinmagazine.com bitcoin OR BTC when:2d"),
    "news_theblock": _google_news_url("site:theblock.co bitcoin OR BTC when:2d"),
    "news_google_btc": _google_news_url("bitcoin OR BTC crypto when:1d"),
}


def _canonical_source(source_name: str) -> str:
    return SOURCE_ALIASES.get(source_name, source_name)


def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("bitcoin", "btc", "crypto", "cryptocurrency", "digital asset"))


def _clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_published(value: object | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    raw = str(value).strip()
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return raw


def _parse_feed(source_name: str, feed: str, timeout: int) -> tuple[list[TextItem], dict]:
    diag = {
        "source": source_name,
        "url": feed,
        "http_status": None,
        "entries": 0,
        "kept": 0,
        "error": None,
    }
    try:
        response = requests.get(feed, headers=HTTP_HEADERS, timeout=timeout)
        diag["http_status"] = response.status_code
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
    except Exception as exc:
        diag["error"] = f"{type(exc).__name__}: {exc}"
        try:
            parsed = feedparser.parse(feed)
        except Exception:
            return [], diag

    entries = list(getattr(parsed, "entries", []) or [])
    diag["entries"] = len(entries)
    rows: list[TextItem] = []
    for item in entries[:75]:
        title = getattr(item, "title", "") or ""
        summary = getattr(item, "summary", "") or getattr(item, "description", "") or ""
        text = _clean_text(" ".join(str(x).strip() for x in [title, summary] if str(x).strip()))
        if not text or not _is_relevant(text):
            continue
        published = (
            getattr(item, "published", None)
            or getattr(item, "updated", None)
            or getattr(item, "created", None)
            or datetime.now(timezone.utc).isoformat()
        )
        rows.append(
            TextItem(
                published_at=_safe_published(published),
                source=_canonical_source(source_name),
                text=text[:2000],
                url=getattr(item, "link", None),
            )
        )
    diag["kept"] = len(rows)
    return rows, diag


def fetch_rss_items(feeds: dict[str, str] | Iterable[str] = RSS_FEEDS, timeout: int = 20) -> list[TextItem]:
    rows: list[TextItem] = []
    items = feeds.items() if isinstance(feeds, dict) else [(f"rss_{i}", feed) for i, feed in enumerate(feeds)]
    for source_name, feed in items:
        parsed_rows, _ = _parse_feed(source_name, feed, timeout)
        rows.extend(parsed_rows)
    return rows


def fetch_gdelt_items(timeout: int = 20) -> list[TextItem]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    rows: list[TextItem] = []
    seen_urls = set()
    for query in ["bitcoin OR BTC", '"bitcoin"', '"BTC" crypto']:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 75,
            "sort": "datedesc",
            "timespan": "48h",
        }
        try:
            data = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout).json()
        except Exception:
            continue
        for art in data.get("articles", []):
            art_url = art.get("url")
            if art_url in seen_urls:
                continue
            text = _clean_text(f"{art.get('title', '')} {art.get('sourceCountry', '')}".strip())
            if text and _is_relevant(text):
                seen_urls.add(art_url)
                rows.append(
                    TextItem(
                        published_at=art.get("seendate", datetime.now(timezone.utc).isoformat()),
                        source="gdelt",
                        text=text[:2000],
                        url=art_url,
                    )
                )
    return rows


def collect_live_text_with_diagnostics() -> tuple[pd.DataFrame, list[dict]]:
    items: list[TextItem] = []
    diagnostics: list[dict] = []

    for feeds, kind in [(RSS_FEEDS, "primary_rss"), (GOOGLE_NEWS_FALLBACK_FEEDS, "google_news_rss")]:
        for source_name, feed in feeds.items():
            rows, diag = _parse_feed(source_name, feed, timeout=20)
            diag["kind"] = kind
            diagnostics.append(diag)
            items.extend(rows)

    gdelt_rows = fetch_gdelt_items()
    diagnostics.append(
        {
            "source": "gdelt",
            "kind": "gdelt_api",
            "url": "https://api.gdeltproject.org/api/v2/doc/doc",
            "kept": len(gdelt_rows),
        }
    )
    items.extend(gdelt_rows)

    if not items:
        df = pd.DataFrame(columns=["published_at", "source", "text", "url"])
    else:
        df = pd.DataFrame([item.__dict__ for item in items]).drop_duplicates(subset=["source", "text"])
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at", "text"])
        df = df.sort_values("published_at")

    counts = df["source"].value_counts().to_dict() if len(df) else {}
    print(f"Live text collection kept {len(df)} rows by source: {counts}")
    for diag in diagnostics:
        print(f"Live text diagnostic: {diag}")
    return df, diagnostics


def collect_live_text() -> pd.DataFrame:
    df, _ = collect_live_text_with_diagnostics()
    return df
