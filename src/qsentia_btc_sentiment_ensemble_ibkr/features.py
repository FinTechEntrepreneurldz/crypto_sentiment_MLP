from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactStore


def _read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp", "unnamed: 0"}), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.columns = [str(c).lower() for c in df.columns]
    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])


def load_price_history(artifacts: ArtifactStore, start: str = "2017-01-01") -> pd.DataFrame:
    hist = _read_ohlcv(artifacts.path("ohlcv.csv"))
    try:
        import yfinance as yf

        live = yf.download("BTC-USD", start=start, auto_adjust=False, progress=False)
        if isinstance(live.columns, pd.MultiIndex):
            live.columns = live.columns.get_level_values(0)
        live.columns = [str(c).lower() for c in live.columns]
        live.index = pd.to_datetime(live.index, utc=True)
        live = live[[c for c in ["open", "high", "low", "close", "volume"] if c in live.columns]]
        live = live.apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])
        hist = pd.concat([hist, live]).sort_index()
        hist = hist[~hist.index.duplicated(keep="last")]
    except Exception:
        pass
    return hist


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return (close.diff(n) / close.shift(n)) * 100.0


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd_sig(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    ema_f = close.ewm(span=fast).mean()
    ema_s = close.ewm(span=slow).mean()
    macd = ema_f - ema_s
    return (macd - macd.ewm(span=sig).mean()) / close


def disc_roc(x: float, std_pct: float) -> str:
    if pd.isna(x) or pd.isna(std_pct) or std_pct == 0:
        return "neutral"
    if x > std_pct:
        return "bullish"
    if x < -std_pct:
        return "bearish"
    return "neutral"


def disc_rsi(x: float) -> str:
    if pd.isna(x):
        return "neutral"
    if x > 70:
        return "bearish"
    if x < 30:
        return "bullish"
    return "neutral"


def technical_row(price_history: pd.DataFrame, artifacts: ArtifactStore) -> tuple[pd.Series, dict[str, Any]]:
    px = price_history["close"].astype(float)
    feat = pd.DataFrame(index=price_history.index)
    feat["ret_1"] = px.pct_change()
    feat["ret_5"] = px.pct_change(5)
    feat["ret_10"] = px.pct_change(10)
    feat["roc_10"] = roc(px, 10)
    feat["rsi_14"] = rsi(px, 14)
    feat["macd_n"] = macd_sig(px)
    feat["vol_8"] = feat["ret_1"].rolling(8).std() * 100.0
    feat["vol_30"] = feat["ret_1"].rolling(30).std() * 100.0
    tbl = artifacts.load_json("live_state/tbl_latest.json")
    feat["sigma"] = float(tbl.get("sigma", 0.0))
    feat["prev_lbl"] = float(tbl.get("label", 0.0))
    row = feat.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).iloc[-1].copy()
    context = {
        "roc_d": disc_roc(float(row["roc_10"]), float(row["vol_8"])),
        "rsi_d": disc_rsi(float(row["rsi_14"])),
        "prev_d": {1.0: "bullish", -1.0: "bearish", 0.0: "neutral"}.get(float(row["prev_lbl"]), "neutral"),
        "price_date": str(price_history.index[-1]),
        "close": float(px.iloc[-1]),
    }
    return row, context


def prompts_from_live_text(text_df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    if text_df.empty:
        return text_df.assign(prompt=pd.Series(dtype=str), day=pd.Series(dtype="datetime64[ns, UTC]"))
    out = text_df.copy()
    out["day"] = pd.to_datetime(out["published_at"], utc=True).dt.normalize()
    out["prompt"] = (
        "Previous Label: "
        + str(context["prev_d"])
        + ", ROC: "
        + str(context["roc_d"])
        + ", RSI: "
        + str(context["rsi_d"])
        + ", Tweet: "
        + out["text"].astype(str)
    )
    return out


def build_live_feature_row(
    artifacts: ArtifactStore,
    price_history: pd.DataFrame,
    scored_text: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    schema = artifacts.load_json("live_state/feature_schema.json")
    feature_cols = list(schema["FEAT_COLS"])
    sources = list(schema["SOURCES"])
    daily = pd.read_parquet(artifacts.path("live_state/daily_features.parquet")).sort_index()
    row = daily[feature_cols].iloc[-1].copy()
    tech, context = technical_row(price_history, artifacts)
    for col, value in tech.items():
        if col in row.index:
            row[col] = float(value)

    for source in sources:
        for suffix in ["p_bear", "p_neut", "p_bull", "n"]:
            col = f"{source}__{suffix}"
            if col in row.index:
                row[col] = 0.0

    if not scored_text.empty:
        for source, group in scored_text.groupby("source"):
            if source not in sources:
                continue
            row[f"{source}__p_bear"] = float(group["p_bear"].mean())
            row[f"{source}__p_neut"] = float(group["p_neut"].mean())
            row[f"{source}__p_bull"] = float(group["p_bull"].mean())
            row[f"{source}__n"] = float(len(group))
        row["all__p_bear"] = float(scored_text["p_bear"].mean())
        row["all__p_neut"] = float(scored_text["p_neut"].mean())
        row["all__p_bull"] = float(scored_text["p_bull"].mean())
        row["all__n"] = float(len(scored_text))
    else:
        for col in ["all__p_bear", "all__p_neut", "all__p_bull", "all__n"]:
            if col in row.index:
                row[col] = 0.0

    row = row.reindex(feature_cols).astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    meta = {"feature_cols": feature_cols, "sources": sources, "context": context}
    return row, meta

