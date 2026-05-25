#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from qsentia_btc_sentiment_ensemble_ibkr.brokers import IBKRBroker
from qsentia_btc_sentiment_ensemble_ibkr.dashboard_logs import write_dashboard_logs
from qsentia_btc_sentiment_ensemble_ibkr.risk import TargetOrder
from qsentia_btc_sentiment_ensemble_ibkr.settings import account_fingerprint, load_settings


def _latest_btc_price(fallback_ohlcv: Path) -> float:
    try:
        import yfinance as yf

        px = yf.download("BTC-USD", period="7d", auto_adjust=False, progress=False)
        close = px["Close"].dropna()
        if len(close):
            return float(close.iloc[-1])
    except Exception:
        pass

    if fallback_ohlcv.exists():
        with fallback_ohlcv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in reversed(rows):
            raw = row.get("close") or row.get("Close")
            if raw:
                return float(raw)
    return 0.0


def main() -> int:
    settings = load_settings()
    broker = IBKRBroker(settings)
    state = broker.state()
    if state.net_liq <= 0:
        raise RuntimeError(
            "IBKR NetLiquidation came back as zero or missing. "
            "Check IBKR_ACCOUNT, TWS/Gateway account value visibility, and paper account connection."
        )

    contract_symbol = settings.ibkr_contract_symbol
    ib = broker._connect()
    try:
        contract = broker._front_mbt_contract(ib)
        contract_symbol = contract.localSymbol or contract.symbol or contract_symbol
    finally:
        ib.disconnect()

    try:
        btc_price = _latest_btc_price(settings.artifact_dir / "ohlcv.csv")
    except Exception:
        btc_price = 0.0

    timestamp = datetime.now(timezone.utc).isoformat()
    signal = SimpleNamespace(
        asof=timestamp,
        signal=0,
        label="FLAT",
        confidence=0.0,
        btc_price=btc_price,
        method="ibkr_account_dashboard_snapshot",
        metadata={
            "research_best_strategy": "Ensemble Dual In-Out",
            "live_text_rows": None,
            "scored_text_rows": None,
            "components": None,
            "snapshot_only": True,
        },
    )
    target = TargetOrder(
        desired_contracts=state.current_contracts,
        current_contracts=state.current_contracts,
        delta_contracts=0,
        action="hold",
        reason="dashboard_snapshot_no_trade",
    )
    order_result = {
        "submitted": False,
        "reason": "dashboard_snapshot_no_trade",
        "contract": contract_symbol,
        "side": "HOLD",
        "qty": 0,
    }
    report = {
        "ts": timestamp,
        "dry_run": False,
        "snapshot_only": True,
        "net_liq": state.net_liq,
        "current_contracts": state.current_contracts,
        "account_fingerprint": account_fingerprint(settings.ibkr_account),
    }

    write_dashboard_logs(
        "logs",
        report=report,
        signal=signal,
        state=state,
        target=target,
        order_result=order_result,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "net_liq": state.net_liq,
                "current_contracts": state.current_contracts,
                "account_fingerprint": account_fingerprint(settings.ibkr_account),
                "contract": contract_symbol,
                "dashboard_logs": "logs",
                "submitted_order": False,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
