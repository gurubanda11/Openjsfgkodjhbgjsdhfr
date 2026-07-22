"""
Small trainable policy network and checkpoint helpers for TSA simulation training.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyCheckpointMetadata:
    """Portable checkpoint metadata for local save/load."""

    version: str
    obs_dim: int
    action_dim: int
    event_name: str
    season: str
    seed: int
    curriculum: str


class DronePolicyNet(nn.Module):
    """Compact MLP used for local TSA sim training and evaluation."""

    def __init__(self, obs_dim: int, action_dim: int = 7, hidden_sizes: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def build_checkpoint_metadata(
    obs_dim: int,
    action_dim: int,
    event_name: str,
    season: str,
    seed: int,
    curriculum: str,
    version: str = "1.0",
) -> PolicyCheckpointMetadata:
    return PolicyCheckpointMetadata(
        version=version,
        obs_dim=obs_dim,
        action_dim=action_dim,
        event_name=event_name,
        season=season,
        seed=seed,
        curriculum=curriculum,
    )


def save_checkpoint(path: Path, model: DronePolicyNet, metadata: PolicyCheckpointMetadata, optimizer: torch.optim.Optimizer | None = None) -> Path:
    """Persist a local checkpoint to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": asdict(metadata),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[DronePolicyNet, PolicyCheckpointMetadata, dict[str, Any]]:
    """Load a local checkpoint and reconstruct the policy network."""

    payload = torch.load(path, map_location=map_location)
    metadata_dict = payload.get("metadata", {})
    metadata = PolicyCheckpointMetadata(
        version=str(metadata_dict.get("version", "1.0")),
        obs_dim=int(metadata_dict.get("obs_dim", 26)),
        action_dim=int(metadata_dict.get("action_dim", 7)),
        event_name=str(metadata_dict.get("event_name", "TSA HS Drone Challenge")),
        season=str(metadata_dict.get("season", "2026")),
        seed=int(metadata_dict.get("seed", 2026)),
        curriculum=str(metadata_dict.get("curriculum", "mixed")),
    )
    model = DronePolicyNet(obs_dim=metadata.obs_dim, action_dim=metadata.action_dim)
    model.load_state_dict(payload["model_state_dict"])
    return model, metadata, payload