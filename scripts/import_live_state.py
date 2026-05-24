#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", help="Zip produced by docs/colab_export_exact_live_state_cell.py")
    parser.add_argument("--artifact-dir", default="artifacts/current")
    args = parser.parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("live_state/"):
                zf.extract(name, artifact_dir)
    print(f"Imported live_state into {artifact_dir.resolve() / 'live_state'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

