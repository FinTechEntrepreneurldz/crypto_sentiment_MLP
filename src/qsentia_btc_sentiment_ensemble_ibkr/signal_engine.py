from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactStore
from .features import build_live_feature_row, load_price_history, prompts_from_live_text
from .model_runtime import attach_scores, generate_component_signals, score_prompts
from .sentiment_live import collect_live_text


@dataclass(frozen=True)
class LiveSignal:
    asof: str
    signal: int
    label: str
    confidence: float
    btc_price: float
    method: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "asof": self.asof,
            "signal": self.signal,
            "label": self.label,
            "confidence": self.confidence,
            "btc_price": self.btc_price,
            "method": self.method,
            "metadata": self.metadata,
        }


def latest_btc_price(fallback_ohlcv: Path | None = None) -> float:
    import yfinance as yf

    try:
        px = yf.download("BTC-USD", period="7d", auto_adjust=False, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        close_col = "Close" if "Close" in px.columns else "close"
        return float(px[close_col].dropna().iloc[-1])
    except Exception:
        if fallback_ohlcv is None or not fallback_ohlcv.exists():
            raise
        hist = pd.read_csv(fallback_ohlcv)
        close_col = "close" if "close" in hist.columns else "Close"
        return float(pd.to_numeric(hist[close_col], errors="coerce").dropna().iloc[-1])


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def live_btc_shock_metrics(price_history: pd.DataFrame, btc_price: float, feature_row: pd.Series) -> dict[str, float | str | None]:
    """Return live shock metrics without changing the model feature row.

    The model itself is daily, but this override needs a real 24h guardrail so
    a sharp intraday BTC selloff is not hidden by stale daily bars.
    """
    metrics: dict[str, float | str | None] = {
        "btc_ret_24h": None,
        "btc_ret_1d": _safe_float(feature_row.get("ret_1")),
        "btc_ret_5d": _safe_float(feature_row.get("ret_5")),
        "btc_rsi_14": _safe_float(feature_row.get("rsi_14")),
        "btc_roc_10": _safe_float(feature_row.get("roc_10")),
        "btc_shock_source": "daily_features_only",
    }
    try:
        import yfinance as yf

        bars = yf.download("BTC-USD", period="3d", interval="1h", auto_adjust=False, progress=False)
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars.index = pd.to_datetime(bars.index, utc=True)
        close_col = "Close" if "Close" in bars.columns else "close"
        closes = pd.to_numeric(bars[close_col], errors="coerce").dropna().sort_index()
        if len(closes) >= 2:
            latest_ts = closes.index[-1]
            cutoff = latest_ts - pd.Timedelta(hours=24)
            past = closes.loc[closes.index <= cutoff]
            past_px = float(past.iloc[-1] if len(past) else closes.iloc[0])
            if past_px > 0:
                metrics["btc_ret_24h"] = float(btc_price / past_px - 1.0)
                metrics["btc_shock_source"] = "yfinance_1h"
    except Exception as exc:
        metrics["btc_shock_source"] = f"daily_features_fallback:{type(exc).__name__}"
    return metrics


def _clearly_directional_components(components: dict, direction: int, block_confidence: float) -> bool:
    checks = [
        ("mlp", components.get("mlp_signal"), components.get("mlp_confidence")),
        ("baseline", components.get("baseline_signal"), components.get("baseline_confidence")),
        ("mean", components.get("mean_signal"), components.get("mean_confidence")),
    ]
    for _name, sig, conf in checks:
        if int(sig or 0) == int(direction) and float(conf or 0.0) >= block_confidence:
            return True
    return False


def apply_btc_shock_override(
    *,
    signal: int,
    confidence: float,
    components: dict,
    shock_metrics: dict,
    text_rows: int,
    min_text_rows: int,
    enabled: bool | None = None,
    ret_24h_threshold: float | None = None,
    ret_5d_threshold: float | None = None,
    rally_ret_24h_threshold: float | None = None,
    rally_ret_5d_threshold: float | None = None,
    confidence_floor: float | None = None,
    rally_confidence_floor: float | None = None,
    bullish_block_confidence: float | None = None,
    bearish_block_confidence: float | None = None,
) -> tuple[int, float, dict]:
    enabled = (
        _bool_env("ENABLE_BTC_SHOCK_OVERRIDE", _bool_env("ENABLE_BTC_CRASH_OVERRIDE", True))
        if enabled is None
        else enabled
    )
    ret_24h_threshold = _float_env("BTC_CRASH_RET_24H_THRESHOLD", -0.025) if ret_24h_threshold is None else ret_24h_threshold
    ret_5d_threshold = _float_env("BTC_CRASH_RET_5D_THRESHOLD", -0.04) if ret_5d_threshold is None else ret_5d_threshold
    rally_ret_24h_threshold = (
        _float_env("BTC_RALLY_RET_24H_THRESHOLD", 0.025)
        if rally_ret_24h_threshold is None
        else rally_ret_24h_threshold
    )
    rally_ret_5d_threshold = (
        _float_env("BTC_RALLY_RET_5D_THRESHOLD", 0.04)
        if rally_ret_5d_threshold is None
        else rally_ret_5d_threshold
    )
    confidence_floor = _float_env("BTC_CRASH_OVERRIDE_CONFIDENCE", 0.12) if confidence_floor is None else confidence_floor
    rally_confidence_floor = (
        _float_env("BTC_RALLY_OVERRIDE_CONFIDENCE", _float_env("BTC_CRASH_OVERRIDE_CONFIDENCE", 0.12))
        if rally_confidence_floor is None
        else rally_confidence_floor
    )
    bullish_block_confidence = (
        _float_env("BTC_CRASH_BULLISH_BLOCK_CONFIDENCE", 0.60)
        if bullish_block_confidence is None
        else bullish_block_confidence
    )
    bearish_block_confidence = (
        _float_env("BTC_RALLY_BEARISH_BLOCK_CONFIDENCE", 0.60)
        if bearish_block_confidence is None
        else bearish_block_confidence
    )

    ret_24h = _safe_float(shock_metrics.get("btc_ret_24h"))
    ret_5d = _safe_float(shock_metrics.get("btc_ret_5d"))
    trigger_24h_down = ret_24h is not None and ret_24h <= ret_24h_threshold
    trigger_5d_down = ret_5d is not None and ret_5d <= ret_5d_threshold
    trigger_24h_up = ret_24h is not None and ret_24h >= rally_ret_24h_threshold
    trigger_5d_up = ret_5d is not None and ret_5d >= rally_ret_5d_threshold

    override_direction = 0
    if trigger_24h_down:
        override_direction = -1
    elif trigger_24h_up:
        override_direction = 1
    elif trigger_5d_down and not trigger_5d_up:
        override_direction = -1
    elif trigger_5d_up and not trigger_5d_down:
        override_direction = 1
    elif trigger_5d_down and trigger_5d_up and ret_5d is not None:
        override_direction = int(np.sign(ret_5d))

    clearly_opposed = (
        _clearly_directional_components(components, 1, bullish_block_confidence)
        if override_direction < 0
        else _clearly_directional_components(components, -1, bearish_block_confidence)
        if override_direction > 0
        else False
    )
    text_ok = int(text_rows) >= int(min_text_rows)
    applied = bool(enabled and text_ok and override_direction != 0 and not clearly_opposed)
    floor = confidence_floor if override_direction < 0 else rally_confidence_floor

    diagnostics = {
        "enabled": bool(enabled),
        "applied": applied,
        "reason": (
            "btc_crash_risk_off"
            if applied and override_direction < 0
            else "btc_rally_risk_on"
            if applied
            else "disabled"
            if not enabled
            else "insufficient_live_text"
            if not text_ok
            else "no_price_shock"
            if override_direction == 0
            else "blocked_by_opposed_components"
        ),
        "pre_override_signal": int(signal),
        "pre_override_confidence": float(confidence),
        "override_direction": int(override_direction),
        "post_override_signal": int(override_direction) if applied else int(signal),
        "post_override_confidence": float(max(confidence, floor)) if applied else float(confidence),
        "ret_24h": ret_24h,
        "ret_5d": ret_5d,
        "ret_24h_threshold": float(ret_24h_threshold),
        "ret_5d_threshold": float(ret_5d_threshold),
        "rally_ret_24h_threshold": float(rally_ret_24h_threshold),
        "rally_ret_5d_threshold": float(rally_ret_5d_threshold),
        "trigger_24h_down": bool(trigger_24h_down),
        "trigger_5d_down": bool(trigger_5d_down),
        "trigger_24h_up": bool(trigger_24h_up),
        "trigger_5d_up": bool(trigger_5d_up),
        "confidence_floor": float(floor),
        "crash_confidence_floor": float(confidence_floor),
        "rally_confidence_floor": float(rally_confidence_floor),
        "bullish_block_confidence": float(bullish_block_confidence),
        "bearish_block_confidence": float(bearish_block_confidence),
        "clearly_opposed_components": bool(clearly_opposed),
        "text_rows": int(text_rows),
        "min_text_rows": int(min_text_rows),
        "shock_source": shock_metrics.get("btc_shock_source"),
    }
    if applied:
        return int(override_direction), float(max(confidence, floor)), diagnostics
    return int(signal), float(confidence), diagnostics


def apply_btc_crash_override(**kwargs) -> tuple[int, float, dict]:
    return apply_btc_shock_override(**kwargs)


class SignalEngine:
    def __init__(
        self,
        artifacts: ArtifactStore,
        allow_approximate: bool = False,
        live_text_min_rows: int = 5,
        allow_low_live_text: bool = False,
    ):
        self.artifacts = artifacts
        self.allow_approximate = allow_approximate
        self.live_text_min_rows = live_text_min_rows
        self.allow_low_live_text = allow_low_live_text

    def generate(self) -> LiveSignal:
        check = self.artifacts.check()
        if not check.research_ok:
            raise RuntimeError(f"Missing required research artifacts: {check.missing_research}")
        if not check.exact_live_ok and not self.allow_approximate:
            raise RuntimeError(
                "Exact live-state artifacts are missing. "
                f"Missing: {check.missing_exact_live}. "
                "Run docs/colab_export_exact_live_state_cell.py in Colab and re-import artifacts, "
                "or set QSENTIA_ALLOW_APPROXIMATE_SIGNAL=true for shadow-only dry runs."
            )

        btc_price = latest_btc_price(self.artifacts.path("ohlcv.csv"))
        metadata = self.artifacts.metadata()

        if check.exact_live_ok:
            feature_schema = self.artifacts.load_json("live_state/feature_schema.json")
            price_history = load_price_history(self.artifacts)
            tech_row, context = build_live_feature_row(self.artifacts, price_history, pd.DataFrame(columns=["source", "p_bear", "p_neut", "p_bull"]))
            text_df = collect_live_text()
            if len(text_df) < self.live_text_min_rows and not self.allow_low_live_text:
                raise RuntimeError(
                    f"Live text gate failed: got {len(text_df)} rows, need at least {self.live_text_min_rows}. "
                    "Set ALLOW_LOW_LIVE_TEXT=true only for diagnostics."
                )
            prompt_df = prompts_from_live_text(text_df, context["context"])
            probs = score_prompts(self.artifacts, prompt_df["prompt"].astype(str).tolist()) if len(prompt_df) else []
            scored = attach_scores(prompt_df, probs)
            feature_row, feature_meta = build_live_feature_row(self.artifacts, price_history, scored)
            components = generate_component_signals(self.artifacts, feature_row)
            components_dict = dict(components.__dict__)
            signal = components.ensemble_signal
            confidence = components.ensemble_confidence
            shock_metrics = live_btc_shock_metrics(price_history, btc_price, feature_row)
            signal, confidence, shock_override = apply_btc_shock_override(
                signal=signal,
                confidence=confidence,
                components=components_dict,
                shock_metrics=shock_metrics,
                text_rows=len(text_df),
                min_text_rows=self.live_text_min_rows,
            )
            components_dict["btc_shock_metrics"] = shock_metrics
            components_dict["btc_shock_override"] = shock_override
            components_dict["btc_crash_override"] = shock_override
            signal_date = datetime.now(timezone.utc).date().isoformat()
            method = "live_scrape_cryptobert_mlp_ppo_ensemble"
            if shock_override.get("applied"):
                method += "_btc_shock_override"
        else:
            # Approximate shadow mode: do not submit real orders from this path.
            text_df = collect_live_text()
            signal = 0
            confidence = 0.0
            method = "approximate_shadow_flat_until_exact_state_exported"
            feature_schema = {}
            signal_date = None
            scored = pd.DataFrame()
            components_dict = None
            feature_meta = {}

        label = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[signal]
        return LiveSignal(
            asof=datetime.now(timezone.utc).isoformat(),
            signal=signal,
            label=label,
            confidence=confidence,
            btc_price=btc_price,
            method=method,
            metadata={
                "research_best_strategy": metadata.get("best_strategy"),
                "research_decision": metadata.get("decision", {}),
                "live_text_rows": int(len(text_df)),
                "scored_text_rows": int(len(scored)),
                "artifact_exact_live_ok": check.exact_live_ok,
                "signal_date": signal_date,
                "feature_count": feature_schema.get("input_dim"),
                "components": components_dict,
                "feature_context": feature_meta.get("context"),
            },
        )


def append_signal_log(signal: LiveSignal, path: str | Path = "logs/signals.jsonl") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signal.to_dict(), default=str) + "\n")
