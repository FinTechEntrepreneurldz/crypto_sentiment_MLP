#!/usr/bin/env python3
from __future__ import annotations

import json

from qsentia_btc_sentiment_ensemble_ibkr.artifacts import ArtifactStore
from qsentia_btc_sentiment_ensemble_ibkr.settings import load_settings


def main() -> int:
    settings = load_settings()
    check = ArtifactStore(settings.artifact_dir).check()
    print(json.dumps(check.__dict__, indent=2, default=str))
    if not check.research_ok:
        return 2
    if not check.exact_live_ok:
        print("\nResearch artifacts are present, but exact live-state artifacts are missing.")
        print("Run docs/colab_export_exact_live_state_cell.py in the Colab and import the new zip.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

