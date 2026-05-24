#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BEST_METADATA = {
    "best_strategy": "Ensemble Dual In-Out",
    "best_model_family": "Ensemble",
    "best_signal_mode": "Dual In-Out",
    "decision": {
        "verdict": "PROCEED_TO_PAPER_TRADING",
        "oos_period": "2023-10-01 -> 2026-05-23",
        "reported_best_ensemble": "Ensemble Dual In-Out",
        "reported_best_ensemble_sharpe": 5.04,
    },
}


def copy_if_exists(src: Path | None, dest: Path) -> None:
    if src is None:
        return
    src = src.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="artifacts/current")
    parser.add_argument("--mlp", default="/Users/lucaszarzeczny/Downloads/mlp.pt")
    parser.add_argument("--ppo0", default="/Users/lucaszarzeczny/Downloads/ppo_seed0.zip")
    parser.add_argument("--ppo1", default="/Users/lucaszarzeczny/Downloads/ppo_seed1.zip")
    parser.add_argument("--ppo2", default="/Users/lucaszarzeczny/Downloads/ppo_seed2.zip")
    parser.add_argument("--config", default="/Users/lucaszarzeczny/Downloads/config.json")
    parser.add_argument("--model", default="/Users/lucaszarzeczny/Downloads/model.safetensors")
    parser.add_argument("--tokenizer-config", default="/Users/lucaszarzeczny/Downloads/tokenizer_config.json")
    parser.add_argument("--tokenizer", default="/Users/lucaszarzeczny/Downloads/tokenizer.json")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    copy_if_exists(Path(args.mlp), dest / "models/mlp.pt")
    copy_if_exists(Path(args.ppo0), dest / "models/ppo/ppo_seed0.zip")
    copy_if_exists(Path(args.ppo1), dest / "models/ppo/ppo_seed1.zip")
    copy_if_exists(Path(args.ppo2), dest / "models/ppo/ppo_seed2.zip")
    copy_if_exists(Path(args.config), dest / "config.json")
    copy_if_exists(Path(args.model), dest / "models/cryptobert_ft/model.safetensors")
    copy_if_exists(Path(args.tokenizer_config), dest / "models/cryptobert_ft/tokenizer_config.json")
    copy_if_exists(Path(args.tokenizer), dest / "models/cryptobert_ft/tokenizer.json")

    metadata = {
        **BEST_METADATA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_import_mode": "loose_files",
        "note": "Loose files import contains model weights but still needs exact live_state export for trading.",
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Imported loose artifacts to {dest.resolve()}")
    print("Next required: import live_state/ from docs/colab_export_exact_live_state_cell.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

