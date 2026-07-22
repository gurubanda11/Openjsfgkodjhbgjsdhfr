"""
Local TSA simulation training entrypoint.

The training routine uses a teacher-student style imitation loop so it is fast enough to
run on a laptop CPU while still producing a checkpoint that can be reloaded locally.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from config import CHECKPOINTS_DIR, DEFAULT_CHECKPOINT_NAME, DEFAULT_SEED, DEFAULT_TRAIN_EPOCHS, DEFAULT_TRAIN_EPISODES
from src.policy import DronePolicyNet, build_checkpoint_metadata, save_checkpoint
from src.sim3d_env import Drone3DSimEnv
from src.tsa_rules import create_default_hs_2026_profile


@dataclass
class TrainingSummary:
    episodes: int
    epochs: int
    dataset_size: int
    final_loss: float
    best_loss: float
    best_score: float
    checkpoint_path: Path
    seed: int
    curriculum: str

    def to_terminal_text(self) -> str:
        return (
            "Training complete\n"
            f"Episodes: {self.episodes}\n"
            f"Epochs: {self.epochs}\n"
            f"Dataset size: {self.dataset_size}\n"
            f"Final loss: {self.final_loss:.6f}\n"
            f"Best loss: {self.best_loss:.6f}\n"
            f"Best score: {self.best_score:.2f}\n"
            f"Checkpoint: {self.checkpoint_path}\n"
        )


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


def collect_expert_dataset(env: Drone3DSimEnv, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_reports: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        observation = env.reset(seed=seed + episode_index, episode_index=episode_index)
        done = False
        step_count = 0
        while not done and step_count < env._step_limit:
            target_action = env.expert_action()
            observations.append(observation.copy())
            targets.append(target_action.copy())
            observation, _, done, info = env.step(target_action)
            step_count += 1
        if env.last_metrics is not None:
            episode_reports.append(env.last_metrics.to_dict())

    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32), episode_reports


def train_policy(
    episodes: int = DEFAULT_TRAIN_EPISODES,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
) -> TrainingSummary:
    """Train a small local policy and save a checkpoint."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    profile = create_default_hs_2026_profile()
    env = Drone3DSimEnv(profile=profile, seed=seed)
    checkpoint_path = checkpoint_path or (CHECKPOINTS_DIR / DEFAULT_CHECKPOINT_NAME)

    observations, targets, episode_reports = collect_expert_dataset(env, episodes=episodes, seed=seed)
    if observations.size == 0:
        raise RuntimeError("No training data was generated from the simulator.")

    model = DronePolicyNet(obs_dim=env.observation_dim, action_dim=env.action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    obs_tensor = _to_tensor(observations)
    target_tensor = _to_tensor(targets)
    dataset_size = int(obs_tensor.shape[0])
    best_loss = float("inf")
    final_loss = 0.0

    for epoch_index in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batch_count = 0
        for batch_start in range(0, dataset_size, 64):
            batch_indices = permutation[batch_start: batch_start + 64]
            batch_obs = obs_tensor[batch_indices]
            batch_target = target_tensor[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_obs)
            loss = loss_fn(prediction, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        final_loss = epoch_loss / max(1, batch_count)
        best_loss = min(best_loss, final_loss)

    best_score = max((float(report.get("score", 0.0)) for report in episode_reports), default=0.0)
    metadata = build_checkpoint_metadata(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        event_name=profile.event_name,
        season=profile.season,
        seed=seed,
        curriculum="mixed",
    )
    save_checkpoint(checkpoint_path, model, metadata, optimizer=optimizer)

    return TrainingSummary(
        episodes=episodes,
        epochs=epochs,
        dataset_size=dataset_size,
        final_loss=final_loss,
        best_loss=best_loss,
        best_score=best_score,
        checkpoint_path=checkpoint_path,
        seed=seed,
        curriculum="mixed",
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the TSA simulation policy")
    parser.add_argument("--episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = train_policy(episodes=args.episodes, epochs=args.epochs, seed=args.seed, checkpoint_path=Path(args.checkpoint) if args.checkpoint else None)
    print(summary.to_terminal_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())