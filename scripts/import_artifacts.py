#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/import_artifacts.py /path/to/best_model_artifacts.zip")
        return 2
    zip_path = Path(sys.argv[1]).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    dest = Path("artifacts/current")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest.parent)
    roots = [p for p in dest.parent.iterdir() if p.is_dir() and p.name.startswith("best_model_")]
    if roots:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(sorted(roots)[-1]), dest)
    print(f"Imported artifacts to {dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

