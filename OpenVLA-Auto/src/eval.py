"""
Evaluation entrypoint for the TSA simulation policy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np
import torch

from config import CHECKPOINTS_DIR, DEFAULT_CHECKPOINT_NAME, DEFAULT_EVAL_EPISODES, DEFAULT_REPORT_NAME, DEFAULT_SEED, REPORTS_DIR
from src.policy import load_checkpoint
from src.sim3d_env import Drone3DSimEnv
from src.tsa_rules import create_default_hs_2026_profile


@dataclass
class EvaluationResult:
    episodes: int
    mean_score: float
    mean_reward: float
    mean_gates_cleared: float
    mean_collisions: float
    mean_time_s: float
    success_rate: float
    report_path: Path
    checkpoint_path: Path

    def to_terminal_text(self) -> str:
        return (
            "Evaluation complete\n"
            f"Episodes: {self.episodes}\n"
            f"Mean score: {self.mean_score:.2f}\n"
            f"Mean reward: {self.mean_reward:.2f}\n"
            f"Mean gates cleared: {self.mean_gates_cleared:.2f}\n"
            f"Mean collisions: {self.mean_collisions:.2f}\n"
            f"Mean time: {self.mean_time_s:.2f}s\n"
            f"Success rate: {self.success_rate:.1%}\n"
            f"Report: {self.report_path}\n"
            f"Checkpoint: {self.checkpoint_path}\n"
        )


def _policy_action(model: torch.nn.Module, observation: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(device)
        action = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    if action.size < 7:
        action = np.pad(action, (0, 7 - action.size))
    return np.clip(action[:7], -1.0, 1.0)


def evaluate_policy(
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
    report_path: Path | None = None,
    headless: bool = True,
) -> EvaluationResult:
    """Run a deterministic evaluation over the local TSA simulator."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    profile = create_default_hs_2026_profile()
    env = Drone3DSimEnv(profile=profile, seed=seed)
    checkpoint_path = checkpoint_path or (CHECKPOINTS_DIR / DEFAULT_CHECKPOINT_NAME)
    report_path = report_path or (REPORTS_DIR / DEFAULT_REPORT_NAME)

    if checkpoint_path.exists():
        model, metadata, _ = load_checkpoint(checkpoint_path, map_location="cpu")
    else:
        model = torch.nn.Sequential(torch.nn.Linear(env.observation_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, env.action_dim), torch.nn.Tanh())
        metadata = None
    model.eval()
    device = torch.device("cpu")

    scores: list[float] = []
    rewards: list[float] = []
    gates: list[int] = []
    collisions: list[int] = []
    times: list[float] = []
    successes: list[bool] = []
    episode_reports: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        observation = env.reset(seed=seed + episode_index, episode_index=episode_index)
        done = False
        total_reward = 0.0
        step_count = 0
        while not done and step_count < env._step_limit:
            if metadata is None:
                action = env.expert_action()
            else:
                action = _policy_action(model, observation, device)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            if not headless:
                frame = env.render()
                import cv2

                cv2.imshow("OpenVLA Auto - TSA Eval", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    done = True
                    break
        metrics = env.last_metrics or env._build_metrics(total_reward, timed_out=True)
        scores.append(float(metrics.score))
        rewards.append(float(metrics.reward))
        gates.append(int(metrics.gates_cleared))
        collisions.append(int(metrics.collisions))
        times.append(float(metrics.time_elapsed_s))
        successes.append(bool(metrics.success))
        episode_reports.append(metrics.to_dict())

    if not headless:
        import cv2

        cv2.destroyAllWindows()

    report = {
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "mean_reward": float(np.mean(rewards)),
        "mean_gates_cleared": float(np.mean(gates)),
        "mean_collisions": float(np.mean(collisions)),
        "mean_time_s": float(np.mean(times)),
        "success_rate": float(np.mean(successes)),
        "episode_reports": episode_reports,
        "checkpoint_path": str(checkpoint_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return EvaluationResult(
        episodes=episodes,
        mean_score=float(report["mean_score"]),
        mean_reward=float(report["mean_reward"]),
        mean_gates_cleared=float(report["mean_gates_cleared"]),
        mean_collisions=float(report["mean_collisions"]),
        mean_time_s=float(report["mean_time_s"]),
        success_rate=float(report["success_rate"]),
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the TSA simulation policy")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = evaluate_policy(
        episodes=args.episodes,
        seed=args.seed,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        report_path=Path(args.report) if args.report else None,
        headless=args.headless,
    )
    print(result.to_terminal_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())