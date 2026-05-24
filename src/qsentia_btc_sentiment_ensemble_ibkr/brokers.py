from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .settings import Settings


@dataclass(frozen=True)
class BrokerState:
    net_liq: float
    current_contracts: int
    raw: dict[str, Any]


class IBKRBroker:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _connect(self):
        from ib_insync import IB

        ib = IB()
        ib.connect(self.settings.ibkr_host, self.settings.ibkr_port, clientId=self.settings.ibkr_client_id)
        return ib

    def _front_mbt_contract(self, ib):
        from ib_insync import Future

        # Empty lastTradeDateOrContractMonth asks IBKR for contract details.
        prototype = Future(
            symbol=self.settings.ibkr_contract_symbol,
            lastTradeDateOrContractMonth="",
            exchange=self.settings.ibkr_exchange,
            currency=self.settings.ibkr_currency,
        )
        details = ib.reqContractDetails(prototype)
        if not details:
            raise RuntimeError("IBKR did not return MBT contract details. Check futures permissions and market data.")
        active = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        return active[0].contract

    def state(self) -> BrokerState:
        ib = self._connect()
        try:
            account = self.settings.ibkr_account
            values = ib.accountValues(account=account) if account else ib.accountValues()
            net_liq = next((float(v.value) for v in values if v.tag == "NetLiquidation" and v.currency == "USD"), 0.0)
            positions = ib.positions(account=account) if account else ib.positions()
            current = 0
            for pos in positions:
                if pos.contract.secType == "FUT" and pos.contract.symbol == self.settings.ibkr_contract_symbol:
                    current += int(pos.position)
            return BrokerState(net_liq=net_liq, current_contracts=current, raw={"positions": len(positions)})
        finally:
            ib.disconnect()

    def submit_delta(self, delta_contracts: int, dry_run: bool) -> dict[str, Any]:
        if delta_contracts == 0:
            return {"submitted": False, "reason": "no_delta"}
        side = "BUY" if delta_contracts > 0 else "SELL"
        qty = abs(delta_contracts)
        if dry_run:
            return {"submitted": False, "dry_run": True, "side": side, "qty": qty}

        from ib_insync import MarketOrder

        ib = self._connect()
        try:
            contract = self._front_mbt_contract(ib)
            order = MarketOrder(side, qty, account=self.settings.ibkr_account or "")
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            return {
                "submitted": True,
                "side": side,
                "qty": qty,
                "contract": contract.localSymbol,
                "order_status": trade.orderStatus.status,
            }
        finally:
            ib.disconnect()


class AlpacaShadowBroker:
    """Long/flat shadow adapter. Alpaca crypto cannot reproduce short BTC exposure."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def state(self) -> BrokerState:
        return BrokerState(net_liq=0.0, current_contracts=0, raw={"warning": "alpaca_shadow_not_exact"})

    def submit_delta(self, delta_contracts: int, dry_run: bool) -> dict[str, Any]:
        return {
            "submitted": False,
            "reason": "alpaca_shadow_adapter_not_exact_for_dual_inout",
            "dry_run": dry_run,
        }

