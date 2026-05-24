from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
            signal = components.ensemble_signal
            confidence = components.ensemble_confidence
            signal_date = datetime.now(timezone.utc).date().isoformat()
            method = "live_scrape_cryptobert_mlp_ppo_ensemble"
        else:
            # Approximate shadow mode: do not submit real orders from this path.
            text_df = collect_live_text()
            signal = 0
            confidence = 0.0
            method = "approximate_shadow_flat_until_exact_state_exported"
            feature_schema = {}
            signal_date = None
            scored = pd.DataFrame()
            components = None
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
                "components": components.__dict__ if components is not None else None,
                "feature_context": feature_meta.get("context"),
            },
        )


def append_signal_log(signal: LiveSignal, path: str | Path = "logs/signals.jsonl") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signal.to_dict(), default=str) + "\n")
