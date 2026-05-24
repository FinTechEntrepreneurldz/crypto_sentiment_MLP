#!/usr/bin/env python3
from __future__ import annotations

import json

from qsentia_btc_sentiment_ensemble_ibkr.brokers import IBKRBroker
from qsentia_btc_sentiment_ensemble_ibkr.settings import load_settings


def main() -> int:
    settings = load_settings()
    broker = IBKRBroker(settings)
    state = broker.state()
    ib = broker._connect()
    try:
        contract = broker._front_mbt_contract(ib)
        report = {
            "connected": True,
            "net_liq": state.net_liq,
            "current_contracts": state.current_contracts,
            "contract": {
                "symbol": contract.symbol,
                "localSymbol": contract.localSymbol,
                "lastTradeDateOrContractMonth": contract.lastTradeDateOrContractMonth,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "conId": contract.conId,
            },
        }
        print(json.dumps(report, indent=2, default=str))
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
