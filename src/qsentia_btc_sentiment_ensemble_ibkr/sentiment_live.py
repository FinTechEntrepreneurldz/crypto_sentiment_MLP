from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import os
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
    # Current live_state feature schema has no native Reddit columns yet.
    # Route Reddit into the trained social BTC bucket until the next retrain.
    "reddit_bitcoin": "hf_btc_tweets",
    "reddit_bitcoinmarkets": "hf_btc_tweets",
    "reddit_cryptocurrency": "hf_btc_tweets",
    "reddit_btc_search": "hf_btc_tweets",
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

REDDIT_RSS_FEEDS = {
    "reddit_bitcoin": "https://www.reddit.com/r/Bitcoin/new/.rss?limit=25",
    "reddit_bitcoinmarkets": "https://www.reddit.com/r/BitcoinMarkets/new/.rss?limit=25",
    "reddit_cryptocurrency": "https://www.reddit.com/r/CryptoCurrency/new/.rss?limit=25",
}

REDDIT_SEARCH_RSS_FEEDS = {
    "reddit_btc_search": "https://www.reddit.com/search.rss?q=bitcoin%20OR%20BTC&sort=new&t=day",
}

YOUTUBE_DEFAULT_QUERIES = [
    ("youtube_btc_news", "bitcoin OR BTC cryptocurrency news"),
    ("youtube_btc_analysis", "bitcoin BTC market analysis"),
]

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def _canonical_source(source_name: str) -> str:
    if source_name.startswith("youtube_"):
        # The current live feature schema has no native YouTube columns yet.
        # Route video titles/descriptions into the trained BTC social bucket.
        return "hf_btc_tweets"
    return SOURCE_ALIASES.get(source_name, source_name)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bounded_int_env(name: str, default: int, lo: int, hi: int) -> int:
    return min(hi, max(lo, _int_env(name, default)))


def _youtube_queries() -> list[tuple[str, str]]:
    raw = os.getenv("YOUTUBE_QUERIES", "").strip()
    if not raw:
        return YOUTUBE_DEFAULT_QUERIES
    queries = [q.strip() for q in raw.split("|") if q.strip()]
    return [(f"youtube_custom_{i + 1}", q) for i, q in enumerate(queries)]


def _max_text_age_hours() -> int:
    return max(1, _int_env("MAX_TEXT_AGE_HOURS", 48))


def _youtube_min_rows() -> int:
    return max(1, _int_env("YOUTUBE_MIN_ROWS", 5))


def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("bitcoin", "btc", "crypto", "cryptocurrency", "digital asset"))


LOW_SIGNAL_PATTERNS = [
    r"\bpromo code\b",
    r"\breferral(?:s| link| code)?\b",
    r"\bdeposit match\b",
    r"\bfor hire\b",
    r"\bfreelance\b",
    r"\bdiscount code\b",
    r"\bgiveaway\b",
    r"\bairdrop\b",
    r"\bpoker\b",
    r"\bcasino\b",
    r"\bonlyfans\b",
    r"\baccept(?:ed|s|ing)? crypto\b",
    r"\bpayment(?:s)? accepted\b",
]


def _is_low_signal_text(text: str) -> bool:
    if not _bool_env("FILTER_LOW_SIGNAL_TEXT", True):
        return False
    t = text.lower()
    return any(re.search(pattern, t) for pattern in LOW_SIGNAL_PATTERNS)


def _dedupe_key(text: str) -> str:
    # Strip syndication/source suffixes so the same headline from multiple feeds
    # does not overweight one story in the daily sentiment row.
    text = text.split(" submitted by ")[0]
    text = re.split(r"\s[-–—]\s", text, maxsplit=1)[0]
    text = re.sub(r"https?://\S+", " ", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())[:180]


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


def _parse_feed(source_name: str, feed: str, timeout: int, max_entries: int = 75) -> tuple[list[TextItem], dict]:
    diag = {
        "source": source_name,
        "canonical_source": _canonical_source(source_name),
        "url": feed,
        "http_status": None,
        "entries": 0,
        "kept": 0,
        "dropped_low_signal": 0,
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
    for item in entries[:max_entries]:
        title = getattr(item, "title", "") or ""
        summary = getattr(item, "summary", "") or getattr(item, "description", "") or ""
        text = _clean_text(" ".join(str(x).strip() for x in [title, summary] if str(x).strip()))
        if not text or not _is_relevant(text):
            continue
        if _is_low_signal_text(text):
            diag["dropped_low_signal"] += 1
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


def fetch_reddit_items(timeout: int = 20) -> tuple[list[TextItem], list[dict]]:
    max_per_feed = _int_env("REDDIT_MAX_PER_FEED", 20)
    max_total = _int_env("REDDIT_MAX_ROWS", 45)
    rows: list[TextItem] = []
    diagnostics: list[dict] = []
    feeds = dict(REDDIT_RSS_FEEDS)
    if _bool_env("REDDIT_ENABLE_SEARCH_FEED", False):
        feeds.update(REDDIT_SEARCH_RSS_FEEDS)
    for source_name, feed in feeds.items():
        parsed_rows, diag = _parse_feed(source_name, feed, timeout=timeout, max_entries=max_per_feed)
        diag["kind"] = "reddit_rss"
        diag["max_entries"] = max_per_feed
        diagnostics.append(diag)
        rows.extend(parsed_rows)
    rows = sorted(rows, key=lambda item: item.published_at)[-max_total:]
    return rows, diagnostics


def fetch_youtube_items(timeout: int = 20) -> tuple[list[TextItem], list[dict]]:
    diagnostics: list[dict] = []
    if not _bool_env("ENABLE_YOUTUBE_API", True):
        return [], [{"source": "youtube", "kind": "youtube_api", "kept": 0, "disabled": True}]

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        return [], [{"source": "youtube", "kind": "youtube_api", "kept": 0, "missing_api_key": True}]

    max_calls = _bounded_int_env("YOUTUBE_MAX_SEARCH_CALLS_PER_RUN", 2, 0, 5)
    max_results = _bounded_int_env("YOUTUBE_MAX_RESULTS_PER_CALL", 25, 1, 50)
    max_rows = _bounded_int_env("YOUTUBE_MAX_ROWS", 40, 1, 250)
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=_max_text_age_hours())
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    rows: list[TextItem] = []
    for source_name, query in _youtube_queries()[:max_calls]:
        start_len = len(rows)
        diag = {
            "source": source_name,
            "canonical_source": _canonical_source(source_name),
            "kind": "youtube_api",
            "query": query,
            "url": YOUTUBE_SEARCH_URL,
            "http_status": None,
            "entries": 0,
            "kept": 0,
            "dropped_low_signal": 0,
            "expected_quota_units": 100,
            "error": None,
        }
        params = {
            "part": "snippet",
            "type": "video",
            "order": "date",
            "maxResults": max_results,
            "q": query,
            "publishedAfter": published_after,
            "relevanceLanguage": "en",
            "regionCode": "US",
            "safeSearch": "none",
            "key": api_key,
        }
        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=params, headers=HTTP_HEADERS, timeout=timeout)
            diag["http_status"] = response.status_code
            if not response.ok:
                diag["error"] = response.text[:500]
                diagnostics.append(diag)
                continue
            data = response.json()
        except Exception as exc:
            diag["error"] = f"{type(exc).__name__}: {exc}"
            diagnostics.append(diag)
            continue

        videos = data.get("items", []) or []
        diag["entries"] = len(videos)
        for video in videos:
            snippet = video.get("snippet", {}) or {}
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            channel = snippet.get("channelTitle", "")
            text = _clean_text(f"{title}. {description}. Channel: {channel}")
            if not text or not _is_relevant(text):
                continue
            if _is_low_signal_text(text):
                diag["dropped_low_signal"] += 1
                continue
            video_id = ((video.get("id", {}) or {}).get("videoId") or "").strip()
            rows.append(
                TextItem(
                    published_at=snippet.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                    source=_canonical_source(source_name),
                    text=text[:2000],
                    url=f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
                )
            )
        diag["kept"] = len(rows) - start_len
        diagnostics.append(diag)

    rows = sorted(rows, key=lambda item: item.published_at)[-max_rows:]
    diagnostics.append(
        {
            "source": "youtube_quota_guard",
            "kind": "quota_guard",
            "search_calls": max_calls,
            "estimated_quota_units": max_calls * 100,
            "max_results_per_call": max_results,
            "max_rows": max_rows,
        }
    )
    return rows, diagnostics


def _enforce_youtube_requirement(rows: list[TextItem], diagnostics: list[dict]) -> None:
    if not _bool_env("REQUIRE_YOUTUBE_LIVE_TEXT", True):
        return
    min_rows = _youtube_min_rows()
    kept = len(rows)
    if kept >= min_rows:
        diagnostics.append(
            {
                "source": "youtube_requirement",
                "kind": "mandatory_source_gate",
                "required": True,
                "min_rows": min_rows,
                "kept": kept,
                "passed": True,
            }
        )
        return

    diagnostics.append(
        {
            "source": "youtube_requirement",
            "kind": "mandatory_source_gate",
            "required": True,
            "min_rows": min_rows,
            "kept": kept,
            "passed": False,
        }
    )
    reason = "YouTube mandatory source gate failed"
    detail = next(
        (
            diag
            for diag in diagnostics
            if diag.get("source") == "youtube"
            or str(diag.get("source", "")).startswith("youtube_")
        ),
        {},
    )
    raise RuntimeError(
        f"{reason}: got {kept} YouTube rows, need at least {min_rows}. "
        f"Check YOUTUBE_API_KEY, ENABLE_YOUTUBE_API, quota, query settings, and recency. "
        f"Diagnostic: {detail}"
    )


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

    if _bool_env("ENABLE_REDDIT_RSS", True):
        reddit_rows, reddit_diagnostics = fetch_reddit_items()
        diagnostics.extend(reddit_diagnostics)
        items.extend(reddit_rows)
    else:
        diagnostics.append({"source": "reddit", "kind": "reddit_rss", "kept": 0, "disabled": True})

    youtube_rows, youtube_diagnostics = fetch_youtube_items()
    _enforce_youtube_requirement(youtube_rows, youtube_diagnostics)
    diagnostics.extend(youtube_diagnostics)
    items.extend(youtube_rows)

    if not items:
        df = pd.DataFrame(columns=["published_at", "source", "text", "url"])
    else:
        df = pd.DataFrame([item.__dict__ for item in items]).drop_duplicates(subset=["source", "text"])
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at", "text"])
        before_recency = len(df)
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - pd.Timedelta(hours=_max_text_age_hours())
        df = df[(df["published_at"] >= cutoff) & (df["published_at"] <= now + pd.Timedelta(hours=6))]
        diagnostics.append(
            {
                "source": "recency_filter",
                "kind": "recency_gate",
                "max_age_hours": _max_text_age_hours(),
                "input_rows": before_recency,
                "kept": int(len(df)),
                "dropped": int(before_recency - len(df)),
            }
        )
        before_dedup = len(df)
        if before_dedup:
            df = df.assign(_dedupe_key=df["text"].astype(str).map(_dedupe_key))
            df = df.sort_values("published_at").drop_duplicates(subset=["_dedupe_key"], keep="last")
            df = df.drop(columns=["_dedupe_key"])
        diagnostics.append(
            {
                "source": "semantic_dedupe",
                "kind": "dedupe_gate",
                "input_rows": before_dedup,
                "kept": int(len(df)),
                "dropped": int(before_dedup - len(df)),
            }
        )
        df = df.sort_values("published_at")

    counts = df["source"].value_counts().to_dict() if len(df) else {}
    print(f"Live text collection kept {len(df)} rows by source: {counts}")
    for diag in diagnostics:
        print(f"Live text diagnostic: {diag}")
    return df, diagnostics


def collect_live_text() -> pd.DataFrame:
    df, _ = collect_live_text_with_diagnostics()
    return df
