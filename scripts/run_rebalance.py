#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qsentia_btc_sentiment_ensemble_ibkr.artifacts import ArtifactStore
from qsentia_btc_sentiment_ensemble_ibkr.brokers import AlpacaShadowBroker, BrokerState, IBKRBroker
from qsentia_btc_sentiment_ensemble_ibkr.risk import target_mbt_contracts
from qsentia_btc_sentiment_ensemble_ibkr.settings import load_settings
from qsentia_btc_sentiment_ensemble_ibkr.signal_engine import SignalEngine, append_signal_log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    dry_run = settings.dry_run or args.dry_run or not args.submit
    artifacts = ArtifactStore(settings.artifact_dir)
    signal = SignalEngine(
        artifacts,
        allow_approximate=settings.allow_approximate_signal,
        live_text_min_rows=settings.live_text_min_rows,
        allow_low_live_text=settings.allow_low_live_text,
    ).generate()
    append_signal_log(signal)

    broker = IBKRBroker(settings) if settings.broker == "ibkr" else AlpacaShadowBroker(settings)
    if dry_run:
        state = BrokerState(
            net_liq=settings.dry_run_net_liq,
            current_contracts=settings.dry_run_current_contracts,
            raw={"offline_dry_run": True},
        )
    else:
        state = broker.state()
    target = target_mbt_contracts(
        signal=signal.signal,
        confidence=signal.confidence,
        net_liq=state.net_liq,
        btc_price=signal.btc_price,
        target_gross_fraction=settings.target_gross_fraction,
        max_contracts=settings.max_contracts,
        allow_short=settings.allow_short,
        min_confidence=settings.min_confidence,
        current_contracts=state.current_contracts,
        tolerance_contracts=settings.rebalance_tolerance_contracts,
    )
    order_result = broker.submit_delta(target.delta_contracts, dry_run=dry_run)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "signal": signal.to_dict(),
        "broker_state": state.__dict__,
        "target": target.__dict__,
        "order_result": order_result,
    }
    out = Path("logs") / f"rebalance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
