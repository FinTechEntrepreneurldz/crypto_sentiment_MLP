from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .brokers import BrokerState
from .risk import TargetOrder

if TYPE_CHECKING:
    from .signal_engine import LiveSignal


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def write_dashboard_logs(
    out_dir: str | Path,
    *,
    report: dict[str, Any],
    signal: LiveSignal,
    state: BrokerState,
    target: TargetOrder,
    order_result: dict[str, Any],
) -> None:
    out = Path(out_dir)
    timestamp_utc = str(report["ts"])
    date_key = timestamp_utc[:10]
    account_status = "dry_run" if report.get("dry_run") else "connected"
    net_liq = float(state.net_liq)
    symbol = str(order_result.get("contract") or "MBT")
    side = str(order_result.get("side") or target.action).upper()
    submitted = bool(order_result.get("submitted"))

    portfolio_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "net_liquidation": net_liq,
        "net_liquidation_value": net_liq,
        "portfolio_value": net_liq,
        "equity": net_liq,
        "account_status": account_status,
        "source": "ibkr_net_liquidation" if not report.get("dry_run") else "dry_run_net_liquidation",
        "current_contracts": state.current_contracts,
        "signal": signal.label,
        "confidence": signal.confidence,
    }
    _write_csv(out / "portfolio" / "portfolio.csv", [portfolio_row], list(portfolio_row), append=True)

    target_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "ticker": symbol,
        "symbol": symbol,
        "target_weight": target.desired_contracts,
        "target_contracts": target.desired_contracts,
        "current_contracts": target.current_contracts,
        "delta_contracts": target.delta_contracts,
        "signal": signal.label,
        "confidence": signal.confidence,
        "btc_price": signal.btc_price,
        "reason": target.reason,
    }
    target_fields = list(target_row)
    _write_csv(out / "target_weights" / "latest_target_weights.csv", [target_row], target_fields)
    _write_csv(out / "target_weights" / "target_weights.csv", [target_row], target_fields, append=True)

    decision_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "action": target.action,
        "signal": signal.label,
        "signal_raw": signal.signal,
        "confidence": signal.confidence,
        "btc_price": signal.btc_price,
        "target_contracts": target.desired_contracts,
        "current_contracts": target.current_contracts,
        "delta_contracts": target.delta_contracts,
        "order_submitted": str(submitted).lower(),
        "portfolio_value": net_liq,
        "net_liquidation": net_liq,
        "account_status": account_status,
        "method": signal.method,
        "live_text_rows": signal.metadata.get("live_text_rows"),
        "scored_text_rows": signal.metadata.get("scored_text_rows"),
        "components_json": _json(signal.metadata.get("components")),
        "order_result_json": _json(order_result),
    }
    decision_fields = list(decision_row)
    _write_csv(out / "decisions" / "latest_decision.csv", [decision_row], decision_fields)
    _write_csv(out / "decisions" / "decisions.csv", [decision_row], decision_fields, append=True)

    position_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "symbol": symbol,
        "qty": target.desired_contracts,
        "current_contracts": target.current_contracts,
        "target_contracts": target.desired_contracts,
        "market_value": target.desired_contracts * float(signal.btc_price) * 0.10,
        "current_price": signal.btc_price,
        "side": "short" if target.desired_contracts < 0 else "long" if target.desired_contracts > 0 else "flat",
    }
    position_fields = list(position_row)
    _write_csv(out / "positions" / "latest_positions.csv", [position_row], position_fields)

    planned_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "symbol": symbol,
        "side": side,
        "qty": abs(target.delta_contracts),
        "delta_contracts": target.delta_contracts,
        "target_contracts": target.desired_contracts,
        "current_contracts": target.current_contracts,
        "submitted": str(submitted).lower(),
        "reason": target.reason,
    }
    planned_fields = list(planned_row)
    _write_csv(out / "orders" / "latest_planned_orders.csv", [planned_row], planned_fields)

    submitted_rows: list[dict[str, Any]] = []
    if submitted:
        submitted_rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "date": date_key,
                "symbol": symbol,
                "side": side,
                "qty": abs(target.delta_contracts),
                "status": order_result.get("order_status"),
                "id": order_result.get("order_id") or "",
                "client_order_id": "",
                "submitted": "true",
            }
        )
    submitted_fields = ["timestamp_utc", "date", "symbol", "side", "qty", "status", "id", "client_order_id", "submitted"]
    _write_csv(out / "orders" / "latest_submitted_orders.csv", submitted_rows, submitted_fields)
    if submitted_rows:
        _write_csv(out / "orders" / "submitted_orders.csv", submitted_rows, submitted_fields, append=True)

    health = {
        "updated_at_utc": timestamp_utc,
        "date": date_key,
        "overall_status": account_status,
        "account_status": account_status,
        "net_liquidation": net_liq,
        "net_liquidation_value": net_liq,
        "portfolio_value": net_liq,
        "equity": net_liq,
        "source": "ibkr_net_liquidation" if not report.get("dry_run") else "dry_run_net_liquidation",
        "model": "Crypto Sentiment MLP/PPO — IBKR",
        "strategy": signal.metadata.get("research_best_strategy"),
        "signal": signal.label,
        "confidence": signal.confidence,
        "target_contracts": target.desired_contracts,
        "current_contracts": target.current_contracts,
        "delta_contracts": target.delta_contracts,
        "submitted_order_count": len(submitted_rows),
        "live_text_rows": signal.metadata.get("live_text_rows"),
        "scored_text_rows": signal.metadata.get("scored_text_rows"),
    }
    (out / "health").mkdir(parents=True, exist_ok=True)
    (out / "health" / "health_status.json").write_text(json.dumps(health, indent=2, default=str), encoding="utf-8")

    signal_row = {
        "timestamp_utc": timestamp_utc,
        "date": date_key,
        "account_status": account_status,
        "net_liquidation": net_liq,
        "portfolio_value": net_liq,
        "signal": signal.label,
        "signal_raw": signal.signal,
        "confidence": signal.confidence,
        "btc_price": signal.btc_price,
        "target_contracts": target.desired_contracts,
        "components_json": _json(signal.metadata.get("components")),
    }
    _write_csv(out / "health" / "signal_history.csv", [signal_row], list(signal_row), append=True)
