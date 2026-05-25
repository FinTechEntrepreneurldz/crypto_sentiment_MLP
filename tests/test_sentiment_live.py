from datetime import datetime, timedelta, timezone

from qsentia_btc_sentiment_ensemble_ibkr.sentiment_live import _canonical_source, _parse_feed


def test_reddit_sources_alias_to_trained_social_bucket():
    assert _canonical_source("reddit_bitcoin") == "hf_btc_tweets"
    assert _canonical_source("reddit_bitcoinmarkets") == "hf_btc_tweets"
    assert _canonical_source("reddit_cryptocurrency") == "hf_btc_tweets"


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
