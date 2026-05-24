from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactStore


@dataclass(frozen=True)
class ComponentSignals:
    baseline_signal: int
    baseline_confidence: float
    mean_signal: int
    mean_confidence: float
    mlp_signal: int
    mlp_confidence: float
    mlp_probs: list[float]
    ppo_signal: int
    ppo_confidence: float
    ppo_position: float
    ppo_actions: list[float]
    ensemble_raw: float
    ensemble_signal: int
    ensemble_confidence: float


def _base_tokenizer_name(artifacts: ArtifactStore) -> str:
    metadata = artifacts.metadata()
    return (
        metadata.get("config", {})
        .get("CONFIG", {})
        .get("MODEL_NAME", "ElKulako/cryptobert")
    )


def _load_tokenizer(artifacts: ArtifactStore):
    from transformers import AutoTokenizer

    model_dir = artifacts.model_dir()
    errors = []
    for location, kwargs in [
        (str(model_dir), {"use_fast": True}),
        (str(model_dir), {"use_fast": False}),
        (_base_tokenizer_name(artifacts), {"use_fast": True}),
        (_base_tokenizer_name(artifacts), {"use_fast": False}),
    ]:
        try:
            return AutoTokenizer.from_pretrained(location, **kwargs)
        except Exception as exc:
            errors.append(f"{location} {kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load any tokenizer. Attempts:\n" + "\n".join(errors))


def score_prompts(artifacts: ArtifactStore, prompts: list[str], batch_size: int = 64) -> np.ndarray:
    if not prompts:
        return np.zeros((0, 3), dtype=np.float32)
    import torch
    from transformers import AutoModelForSequenceClassification

    model_dir = artifacts.model_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = _load_tokenizer(artifacts)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs.extend(torch.softmax(model(**enc).logits, dim=-1).detach().cpu().tolist())
    return np.asarray(probs, dtype=np.float32)


def attach_scores(text_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = text_df.copy()
    if len(out) == 0:
        return out.assign(p_bear=[], p_neut=[], p_bull=[])
    out["p_bear"] = probs[:, 0]
    out["p_neut"] = probs[:, 1]
    out["p_bull"] = probs[:, 2]
    return out


def _baseline_from_row(row: pd.Series, min_messages: int = 1) -> tuple[int, float, int, float]:
    p = np.array([row.get("all__p_bear", 0.0), row.get("all__p_neut", 0.0), row.get("all__p_bull", 0.0)], dtype=float)
    n = float(row.get("all__n", 0.0))
    cls = int(np.argmax(p)) if p.sum() > 0 else 1
    maj_sig = {0: -1, 1: 0, 2: 1}[cls]
    if n < min_messages:
        maj_sig = 0
    maj_conf = float(np.max(p)) if p.size else 0.0
    enc = 2.0 * float(row.get("all__p_bull", 0.0)) + 1.0 * float(row.get("all__p_neut", 0.0))
    mean_sig = 1 if enc > 1.55 else -1 if enc < 0.45 else 0
    if n < min_messages:
        mean_sig = 0
    mean_conf = float(abs(enc - 1.0))
    return int(maj_sig), maj_conf, int(mean_sig), mean_conf


def _load_mlp(artifacts: ArtifactStore, input_dim: int):
    import torch
    import torch.nn as nn

    class MetaMLP(nn.Module):
        def __init__(self, n_in: int, n_out: int = 3, h: int = 128, p: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, h),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(h, h),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(h, n_out),
            )

        def forward(self, x):
            return self.net(x)

    model = MetaMLP(input_dim)
    state = torch.load(artifacts.path("models/mlp.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _mlp_signal(artifacts: ArtifactStore, x: np.ndarray) -> tuple[int, float, list[float]]:
    import torch

    model = _load_mlp(artifacts, len(x))
    with torch.no_grad():
        x_tensor = torch.tensor(x.astype(float).tolist(), dtype=torch.float32).unsqueeze(0)
        probs = torch.softmax(model(x_tensor), dim=-1).detach().cpu().squeeze(0).tolist()
    cls = int(np.argmax(probs))
    return {0: -1, 1: 0, 2: 1}[cls], float(max(probs)), [float(v) for v in probs]


def _patch_numpy_core_aliases() -> None:
    # PPO artifacts were exported from a NumPy 2 runtime, whose pickle paths use
    # numpy._core.*. The Intel Mac paper runner uses NumPy 1.26 for Torch wheel
    # compatibility, where the same modules live under numpy.core.*.
    import sys
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


def _ppo_signal(artifacts: ArtifactStore, x: np.ndarray) -> tuple[int, float, float, list[float]]:
    _patch_numpy_core_aliases()
    import gymnasium as gym
    from stable_baselines3 import PPO

    obs = np.concatenate([x.astype(np.float32), np.array([0.0, 0.0], dtype=np.float32)])
    custom_objects = {
        # Avoid unpickling training-time Gym spaces/RNG objects saved from a
        # different NumPy runtime. Prediction only needs compatible spaces.
        "observation_space": gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32),
        "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": [],
        "ep_success_buffer": [],
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0003,
    }
    actions = []
    for seed in [0, 1, 2]:
        path = artifacts.path(f"models/ppo/ppo_seed{seed}.zip")
        model = PPO.load(str(path), device="cpu", custom_objects=custom_objects)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(np.clip(action[0], -1.0, 1.0)))
    pos = float(np.mean(actions))
    return int(np.sign(pos)), abs(pos), pos, actions


def generate_component_signals(artifacts: ArtifactStore, feature_row: pd.Series) -> ComponentSignals:
    weights = artifacts.load_json("live_state/ensemble_weights.json")["weights"]
    norm = np.load(artifacts.path("live_state/normalization.npz"))
    mu = norm["mu"].astype(np.float32)
    sd = norm["sd"].astype(np.float32)
    sd = np.where(sd == 0, 1.0, sd)
    x = ((feature_row.to_numpy(dtype=np.float32) - mu) / sd).astype(np.float32)
    baseline_sig, baseline_conf, mean_sig, mean_conf = _baseline_from_row(feature_row)
    mlp_sig, mlp_conf, mlp_probs = _mlp_signal(artifacts, x)
    ppo_sig, ppo_conf, ppo_pos, ppo_actions = _ppo_signal(artifacts, x)
    raw = (
        float(weights.get("Majority", 0.0)) * baseline_sig * baseline_conf
        + float(weights.get("Mean", 0.0)) * mean_sig * mean_conf
        + float(weights.get("MLP", 0.0)) * mlp_sig * mlp_conf
        + float(weights.get("PPO", 0.0)) * ppo_sig * ppo_conf
    )
    return ComponentSignals(
        baseline_signal=baseline_sig,
        baseline_confidence=baseline_conf,
        mean_signal=mean_sig,
        mean_confidence=mean_conf,
        mlp_signal=mlp_sig,
        mlp_confidence=mlp_conf,
        mlp_probs=mlp_probs,
        ppo_signal=ppo_sig,
        ppo_confidence=ppo_conf,
        ppo_position=ppo_pos,
        ppo_actions=ppo_actions,
        ensemble_raw=raw,
        ensemble_signal=int(np.sign(raw)),
        ensemble_confidence=float(min(abs(raw), 1.0)),
    )
