from datetime import datetime, timedelta, timezone

from qsentia_btc_sentiment_ensemble_ibkr import sentiment_live
from qsentia_btc_sentiment_ensemble_ibkr.sentiment_live import (
    _canonical_source,
    _dedupe_key,
    _enforce_youtube_requirement,
    fetch_youtube_items,
    _is_low_signal_text,
    _parse_feed,
)


def test_reddit_sources_alias_to_trained_social_bucket():
    assert _canonical_source("reddit_bitcoin") == "hf_btc_tweets"
    assert _canonical_source("reddit_bitcoinmarkets") == "hf_btc_tweets"
    assert _canonical_source("reddit_cryptocurrency") == "hf_btc_tweets"


def test_youtube_sources_alias_to_trained_social_bucket():
    assert _canonical_source("youtube_btc_news") == "hf_btc_tweets"
    assert _canonical_source("youtube_custom_1") == "hf_btc_tweets"


def test_parse_reddit_like_feed_routes_to_social_bucket():
    published = datetime.now(timezone.utc) - timedelta(hours=1)
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Bitcoin market structure improving</title>
          <description>BTC liquidity and crypto sentiment look constructive.</description>
          <pubDate>{published}</pubDate>
          <link>https://www.reddit.com/r/Bitcoin/comments/example</link>
        </item>
      </channel>
    </rss>
    """.format(published=published.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    rows, diag = _parse_feed("reddit_bitcoin", feed, timeout=1, max_entries=5)
    assert diag["canonical_source"] == "hf_btc_tweets"
    assert diag["kept"] == 1
    assert rows[0].source == "hf_btc_tweets"
    assert "Bitcoin market structure" in rows[0].text


def test_low_signal_reddit_spam_is_filtered():
    published = datetime.now(timezone.utc) - timedelta(hours=1)
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Get a Coinbase referral bonus paid in BTC</title>
          <description>Use my referral link and promo code for a deposit match.</description>
          <pubDate>{published}</pubDate>
          <link>https://www.reddit.com/r/referralcodes/comments/example</link>
        </item>
      </channel>
    </rss>
    """.format(published=published.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    rows, diag = _parse_feed("reddit_bitcoin", feed, timeout=1, max_entries=5)
    assert rows == []
    assert diag["dropped_low_signal"] == 1


def test_duplicate_news_key_removes_publisher_suffix():
    assert _dedupe_key("Bitcoin holds support - Yahoo Finance") == _dedupe_key(
        "Bitcoin holds support - AOL.com"
    )
    assert _is_low_signal_text("BTC poker deposit match referral code")


def test_youtube_disabled_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("ENABLE_YOUTUBE_API", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    rows, diagnostics = fetch_youtube_items(timeout=1)
    assert rows == []
    assert diagnostics[0]["disabled"] is True


def test_youtube_missing_key_is_diagnostic_not_failure(monkeypatch):
    monkeypatch.setenv("ENABLE_YOUTUBE_API", "true")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    rows, diagnostics = fetch_youtube_items(timeout=1)
    assert rows == []
    assert diagnostics[0]["missing_api_key"] is True


def test_youtube_requirement_fails_when_missing(monkeypatch):
    monkeypatch.setenv("REQUIRE_YOUTUBE_LIVE_TEXT", "true")
    monkeypatch.setenv("YOUTUBE_MIN_ROWS", "2")
    diagnostics = [{"source": "youtube", "kind": "youtube_api", "kept": 0, "missing_api_key": True}]
    try:
        _enforce_youtube_requirement([], diagnostics)
    except RuntimeError as exc:
        assert "YouTube mandatory source gate failed" in str(exc)
    else:
        raise AssertionError("Expected mandatory YouTube gate failure")
    assert diagnostics[-1]["source"] == "youtube_requirement"
    assert diagnostics[-1]["passed"] is False


def test_youtube_requirement_can_be_disabled_for_diagnostics(monkeypatch):
    monkeypatch.setenv("REQUIRE_YOUTUBE_LIVE_TEXT", "false")
    diagnostics = [{"source": "youtube", "kind": "youtube_api", "kept": 0}]
    _enforce_youtube_requirement([], diagnostics)
    assert diagnostics == [{"source": "youtube", "kind": "youtube_api", "kept": 0}]


def test_youtube_search_rows_are_capped_and_routed(monkeypatch):
    class Response:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {
                "items": [
                    {
                        "id": {"videoId": "abc123"},
                        "snippet": {
                            "publishedAt": datetime.now(timezone.utc).isoformat(),
                            "title": "Bitcoin market update",
                            "description": "BTC liquidity and crypto market structure.",
                            "channelTitle": "QSentia Test",
                        },
                    }
                ]
            }

    def fake_get(url, params, headers, timeout):
        assert url == sentiment_live.YOUTUBE_SEARCH_URL
        assert params["part"] == "snippet"
        assert params["type"] == "video"
        assert params["maxResults"] == 25
        assert params["key"] == "test-key"
        return Response()

    monkeypatch.setenv("ENABLE_YOUTUBE_API", "true")
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    monkeypatch.setenv("YOUTUBE_MAX_SEARCH_CALLS_PER_RUN", "1")
    monkeypatch.setattr(sentiment_live.requests, "get", fake_get)

    rows, diagnostics = fetch_youtube_items(timeout=1)
    assert len(rows) == 1
    assert rows[0].source == "hf_btc_tweets"
    assert rows[0].url == "https://www.youtube.com/watch?v=abc123"
    assert diagnostics[0]["kept"] == 1
    assert diagnostics[-1]["estimated_quota_units"] == 100
