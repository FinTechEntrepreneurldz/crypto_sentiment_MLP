from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_RESEARCH_FILES = [
    "metadata.json",
    "config.json",
    "models/cryptobert_ft/config.json",
    "models/cryptobert_ft/model.safetensors",
    "models/cryptobert_ft/tokenizer.json",
    "models/mlp.pt",
    "models/ppo/ppo_seed0.zip",
    "models/ppo/ppo_seed1.zip",
    "models/ppo/ppo_seed2.zip",
    "ohlcv.csv",
    "results/oos_v3.csv",
]

REQUIRED_EXACT_LIVE_FILES = [
    "live_state/feature_schema.json",
    "live_state/normalization.npz",
    "live_state/ensemble_weights.json",
    "live_state/tbl_latest.json",
]


@dataclass(frozen=True)
class ArtifactCheck:
    root: Path
    research_ok: bool
    exact_live_ok: bool
    missing_research: list[str]
    missing_exact_live: list[str]

    @property
    def tradable(self) -> bool:
        return self.research_ok and self.exact_live_ok


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def load_json(self, rel: str) -> dict[str, Any]:
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def check(self) -> ArtifactCheck:
        missing_research = [rel for rel in REQUIRED_RESEARCH_FILES if not self.path(rel).exists()]
        missing_live = [rel for rel in REQUIRED_EXACT_LIVE_FILES if not self.path(rel).exists()]
        return ArtifactCheck(
            root=self.root,
            research_ok=not missing_research,
            exact_live_ok=not missing_live,
            missing_research=missing_research,
            missing_exact_live=missing_live,
        )

    def metadata(self) -> dict[str, Any]:
        return self.load_json("metadata.json")

    def model_dir(self) -> Path:
        return self.path("models/cryptobert_ft")

