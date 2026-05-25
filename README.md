# QSentia BTC Sentiment Ensemble IBKR

Production paper-trading scaffold for the BTC sentiment ensemble research hit.

Recommended GitHub repo name:

`qsentia-btc-sentiment-ensemble-ibkr`

## Broker Decision

Use **Interactive Brokers** for the real paper-trade target.

The selected strategy is `Ensemble Dual In-Out`, which can flip long or short. Alpaca crypto is useful as a long/flat shadow test, but it cannot reproduce the best strategy because Alpaca crypto is spot, non-marginable, and not shortable. IBKR can express long/short BTC exposure through regulated CME Micro Bitcoin futures (`MBT`, 0.10 BTC per contract), subject to futures/crypto-futures permissions and margin.

## Artifact Status

The supplied bundle contains:

- fine-tuned CryptoBERT model
- MLP weights
- PPO seed models
- OHLCV
- OOS reports
- metadata/config

The bundle is missing a few live-trading-critical objects:

- `feature_schema.json`: exact `FEAT_COLS`, source order, model input size
- `normalization.npz`: exact MLP `mu` and `sd`
- `ensemble_weights.json`: validation-selected Majority/Mean/MLP/PPO weights
- `tbl_latest.json`: current TBL barrier params
- current `messages_v3.parquet` or daily feature cache

Run `scripts/verify_artifacts.py` after importing artifacts. If it reports missing exact-live state, rerun the export patch cell in `docs/colab_export_exact_live_state_cell.py` inside the research Colab and re-import the new zip.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[nlp]"
cp config/paper.example.env .env
python scripts/import_artifacts.py /path/to/best_model_artifacts_20260524_125657_utc.zip
python scripts/import_live_state_files.py  # if the live_state files are in ~/Downloads
python scripts/verify_artifacts.py
python scripts/run_rebalance.py --dry-run
```

To submit paper orders through IBKR:

```bash
python scripts/run_rebalance.py --submit
```

Keep TWS or IB Gateway open and logged into the paper account.

## Self-Hosted Runner Artifact Setup

Do not commit the model files to GitHub. `model.safetensors` is roughly 500MB and should live on the self-hosted runner or in release/storage, then be imported at runtime.

Recommended runner variables:

```text
QSENTIA_ARTIFACT_DIR=/Users/lucaszarzeczny/Documents/QSentia/crypto_sentiment_MLP/artifacts/current
QSENTIA_ARTIFACT_ZIP=/Users/lucaszarzeczny/Downloads/best_model_artifacts_20260524_125657_utc.zip
```

If you export live state as a zip from Colab, also set:

```text
QSENTIA_LIVE_STATE_ZIP=/path/to/exact_live_state_*.zip
```

Or manually copy the live state files into:

```text
artifacts/current/live_state/
```

## Live Pipeline

Daily paper-trade flow:

1. Pull recent BTC price data.
2. Pull live no-auth sentiment sources: CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, The Block, GDELT, and capped Reddit RSS.
3. Build the exact context-aware prompt used in the notebook: previous TBL label, ROC state, RSI state, and article text.
4. Score live text with the exported fine-tuned CryptoBERT.
5. Rebuild the exact 50-column daily feature row from `feature_schema.json`.
6. Apply the exported MLP model and the three exported PPO agents.
7. Blend Majority/Mean/MLP/PPO using `ensemble_weights.json`.
8. Convert `Ensemble Dual In-Out` signal to target MBT contracts.
9. Submit IBKR paper order, or log dry-run ticket.

The submit path enforces a live text gate. By default it requires at least five fresh text rows:

```bash
LIVE_TEXT_MIN_ROWS=5
ALLOW_LOW_LIVE_TEXT=false
```

Only set `ALLOW_LOW_LIVE_TEXT=true` for diagnostics, never for a real paper-trade schedule.

Reddit is enabled by default through public RSS feeds for `r/Bitcoin`, `r/BitcoinMarkets`, and `r/CryptoCurrency`. Because the current model was trained before Reddit had its own feature columns, Reddit posts are scored with the same CryptoBERT pipeline and routed into the existing `hf_btc_tweets` social bucket plus the global `all__*` aggregate. Retrain later if you want Reddit to become a first-class source bucket.

```bash
ENABLE_REDDIT_RSS=true
REDDIT_MAX_PER_FEED=20
REDDIT_MAX_ROWS=45
REDDIT_ENABLE_SEARCH_FEED=false
MAX_TEXT_AGE_HOURS=48
```

## Safety

The default is `DRY_RUN=true`. The bot refuses to trade when exact live-state artifacts are missing unless `QSENTIA_ALLOW_APPROXIMATE_SIGNAL=true` is explicitly set.

## Sizing

When `DRY_RUN=false`, position sizing uses the current IBKR `NetLiquidation` value pulled at runtime, not a fixed starting balance. The default production-paper setting targets up to 90% gross exposure before confidence scaling:

```text
TARGET_GROSS_FRACTION=0.90
MAX_CONTRACTS=150
```

For example, with `$1,000,000` net liquidation and model confidence `0.60`, the target notional is roughly `$540,000`. As the paper account grows or shrinks, the next workflow run recalculates from the updated IBKR net liquidation value.
