#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from qsentia_btc_sentiment_ensemble_ibkr.sentiment_live import collect_live_text_with_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--youtube-min-rows", type=int, default=None)
    args = parser.parse_args()

    df, diagnostics = collect_live_text_with_diagnostics()
    payload = {
        "rows": int(len(df)),
        "min_rows": args.min_rows,
        "by_source": df["source"].value_counts().to_dict() if len(df) else {},
        "sample": df.tail(10).to_dict(orient="records") if len(df) else [],
        "diagnostics": diagnostics,
    }
    print(json.dumps(payload, indent=2, default=str))
    if len(df) < args.min_rows:
        raise RuntimeError(f"Live text source check failed: got {len(df)} rows, need at least {args.min_rows}")
    if args.youtube_min_rows is not None:
        youtube_kept = sum(
            int(diag.get("kept", 0) or 0)
            for diag in diagnostics
            if str(diag.get("kind")) == "youtube_api"
            and str(diag.get("source", "")).startswith("youtube_")
        )
        if youtube_kept < args.youtube_min_rows:
            raise RuntimeError(
                f"YouTube source check failed: got {youtube_kept} rows, need at least {args.youtube_min_rows}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
