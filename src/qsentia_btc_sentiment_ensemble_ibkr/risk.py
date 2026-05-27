from __future__ import annotations

from dataclasses import dataclass


MBT_BTC_MULTIPLIER = 0.10


@dataclass(frozen=True)
class TargetOrder:
    desired_contracts: int
    current_contracts: int
    delta_contracts: int
    action: str
    reason: str


def target_mbt_contracts(
    signal: int,
    confidence: float,
    net_liq: float,
    btc_price: float,
    target_gross_fraction: float,
    max_contracts: int,
    allow_short: bool,
    min_confidence: float,
    current_contracts: int = 0,
    tolerance_contracts: int = 0,
    flatten_on_low_confidence: bool = False,
) -> TargetOrder:
    if confidence < min_confidence:
        if current_contracts != 0 and not flatten_on_low_confidence:
            desired = current_contracts
            reason = "hold_existing_low_confidence"
        else:
            desired = 0
            reason = "flat_or_low_confidence"
    elif signal == 0:
        desired = 0
        reason = "flat_signal"
    elif signal < 0 and not allow_short:
        desired = 0
        reason = "short_signal_blocked"
    else:
        notional = max(net_liq, 0.0) * max(target_gross_fraction, 0.0) * min(confidence, 1.0)
        one_contract_notional = max(btc_price * MBT_BTC_MULTIPLIER, 1.0)
        contracts = int(round(notional / one_contract_notional))
        contracts = max(1, contracts) if notional > 0 else 0
        contracts = min(contracts, max_contracts)
        desired = contracts if signal > 0 else -contracts
        reason = "target_from_signal"

    delta = desired - current_contracts
    if abs(delta) <= tolerance_contracts:
        delta = 0
        action = "hold"
    elif delta > 0:
        action = "buy"
    else:
        action = "sell"
    return TargetOrder(
        desired_contracts=desired,
        current_contracts=current_contracts,
        delta_contracts=delta,
        action=action,
        reason=reason,
    )
