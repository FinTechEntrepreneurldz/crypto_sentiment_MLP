# Paste this cell into the Colab after the model/backtest cells and rerun the artifact export.
# It adds the missing exact-live state needed by the production paper trader.

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_STATE_DIR = PROJECT_DIR / "live_state"
LIVE_STATE_DIR.mkdir(parents=True, exist_ok=True)

feature_schema = {
    "FEAT_COLS": list(FEAT_COLS),
    "SOURCES": list(SOURCES),
    "input_dim": int(len(FEAT_COLS)),
    "model": "MetaMLP",
    "target_map": {"bear": 0, "neutral": 1, "bull": 2},
}
(LIVE_STATE_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

np.savez(
    LIVE_STATE_DIR / "normalization.npz",
    mu=np.asarray(mu, dtype=np.float32),
    sd=np.asarray(sd, dtype=np.float32),
)

ensemble_weights = {
    "weights": {k: float(v) for k, v in weights.items()},
    "last_known_signal": int(ENS["signal"].iloc[-1]),
    "last_known_confidence": float(ENS["confidence"].iloc[-1]),
    "last_signal_date": str(ENS.index[-1]),
}
(LIVE_STATE_DIR / "ensemble_weights.json").write_text(json.dumps(ensemble_weights, indent=2), encoding="utf-8")

last_tbl = TBL.iloc[-1].to_dict()
(LIVE_STATE_DIR / "tbl_latest.json").write_text(json.dumps({k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in last_tbl.items()}, indent=2), encoding="utf-8")

DAILY.to_parquet(LIVE_STATE_DIR / "daily_features.parquet")
BASE_SIG.to_parquet(LIVE_STATE_DIR / "baseline_signal.parquet")
ENS.to_parquet(LIVE_STATE_DIR / "ensemble_signal.parquet")
MSG_CTX.tail(10000).to_parquet(LIVE_STATE_DIR / "recent_scored_messages.parquet")

export_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
zip_path = PROJECT_DIR / f"exact_live_state_{export_id}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in LIVE_STATE_DIR.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(PROJECT_DIR))

print("Exact live-state export complete:", zip_path)

