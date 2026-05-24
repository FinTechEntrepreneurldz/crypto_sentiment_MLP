#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULTS = {
    "feature_schema.json": "/Users/lucaszarzeczny/Downloads/feature_schema.json",
    "normalization.npz": "/Users/lucaszarzeczny/Downloads/normalization.npz",
    "ensemble_weights.json": "/Users/lucaszarzeczny/Downloads/ensemble_weights.json",
    "tbl_latest.json": "/Users/lucaszarzeczny/Downloads/tbl_latest.json",
    "daily_features.parquet": "/Users/lucaszarzeczny/Downloads/daily_features.parquet",
    "baseline_signal.parquet": "/Users/lucaszarzeczny/Downloads/baseline_signal.parquet",
    "ensemble_signal.parquet": "/Users/lucaszarzeczny/Downloads/ensemble_signal.parquet",
    "recent_scored_messages.parquet": "/Users/lucaszarzeczny/Downloads/recent_scored_messages.parquet",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/current")
    args = parser.parse_args()
    live_dir = Path(args.artifact_dir) / "live_state"
    live_dir.mkdir(parents=True, exist_ok=True)
    for name, src in DEFAULTS.items():
        source = Path(src).expanduser()
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, live_dir / name)
    print(f"Imported live-state files into {live_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

