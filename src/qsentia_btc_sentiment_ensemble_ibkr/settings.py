from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Keep verification usable before dependencies are installed.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def account_fingerprint(account: str | None) -> str:
    if not account:
        return ""
    return hashlib.sha256(account.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Settings:
    broker: str
    dry_run: bool
    artifact_dir: Path
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    ibkr_account: str | None
    ibkr_exchange: str
    ibkr_contract_symbol: str
    ibkr_currency: str
    target_gross_fraction: float
    max_contracts: int
    min_confidence: float
    flatten_on_low_confidence: bool
    rebalance_tolerance_contracts: int
    allow_short: bool
    allow_approximate_signal: bool
    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_paper: bool
    alpaca_symbol: str
    dry_run_net_liq: float
    dry_run_current_contracts: int
    live_text_min_rows: int
    allow_low_live_text: bool


def load_settings(env_path: str | Path = ".env") -> Settings:
    load_dotenv(env_path, override=False)
    return Settings(
        broker=os.getenv("BROKER", "ibkr").lower(),
        dry_run=_bool("DRY_RUN", True),
        artifact_dir=Path(os.getenv("QSENTIA_ARTIFACT_DIR", "artifacts/current")),
        ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.getenv("IBKR_PORT", "7497")),
        ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID", "37")),
        ibkr_account=os.getenv("IBKR_ACCOUNT") or None,
        ibkr_exchange=os.getenv("IBKR_EXCHANGE", "CME"),
        ibkr_contract_symbol=os.getenv("IBKR_CONTRACT_SYMBOL", "MBT"),
        ibkr_currency=os.getenv("IBKR_CURRENCY", "USD"),
        target_gross_fraction=float(os.getenv("TARGET_GROSS_FRACTION", "0.90")),
        max_contracts=int(os.getenv("MAX_CONTRACTS", "150")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.10")),
        flatten_on_low_confidence=_bool("FLATTEN_ON_LOW_CONFIDENCE", False),
        rebalance_tolerance_contracts=int(os.getenv("REBALANCE_TOLERANCE_CONTRACTS", "0")),
        allow_short=_bool("ALLOW_SHORT", True),
        allow_approximate_signal=_bool("QSENTIA_ALLOW_APPROXIMATE_SIGNAL", False),
        alpaca_api_key=os.getenv("ALPACA_API_KEY") or None,
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY") or None,
        alpaca_paper=_bool("ALPACA_PAPER", True),
        alpaca_symbol=os.getenv("ALPACA_SYMBOL", "BTC/USD"),
        dry_run_net_liq=float(os.getenv("DRY_RUN_NET_LIQ", "100000")),
        dry_run_current_contracts=int(os.getenv("DRY_RUN_CURRENT_CONTRACTS", "0")),
        live_text_min_rows=int(os.getenv("LIVE_TEXT_MIN_ROWS", "5")),
        allow_low_live_text=_bool("ALLOW_LOW_LIVE_TEXT", False),
    )
