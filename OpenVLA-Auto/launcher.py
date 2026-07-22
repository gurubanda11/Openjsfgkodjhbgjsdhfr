"""
Master launcher for OpenVLA Auto.
Supports live webcam inference and the TSA simulation / training / evaluation pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from config import DEFAULT_LAUNCHER_MODE, DEFAULT_SEED
from installer import DependencyInstaller, VenvManager
from src.device import detect_system_info
from src.utils import errors_logger, install_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVLA Auto launcher")
    parser.add_argument("mode", nargs="?", default=DEFAULT_LAUNCHER_MODE, choices=["live", "simulate", "train", "eval"], help="Launcher workflow")
    parser.add_argument("--headless", action="store_true", help="Run simulation/evaluation without opening windows")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency auto-installation")
    parser.add_argument("--no-model-download", action="store_true", help="Skip OpenVLA model download/load in live mode")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes for simulate/train/eval")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for train/eval")
    parser.add_argument("--report", type=str, default=None, help="Evaluation report output path")
    return parser


def _run_live_mode(skip_install: bool, skip_model_download: bool) -> int:
    print("\n" + "=" * 50)
    print("           WELCOME TO OPENVLA AUTO          ")
    print("=" * 50 + "\n")

    if not VenvManager.is_in_venv():
        print("Initializing Virtual Environment...")
        VenvManager.activate_or_reexec()

    sys_info = detect_system_info()
    print(sys_info.format_terminal_ui())
    print("\n")

    if not skip_install:
        print("Checking system dependencies...")
        installer = DependencyInstaller(system_info=sys_info)
        if not installer.run_installation():
            print("\nFatal: Dependency installation failed. Check logs for details.")
            return 1

    from src.downloader import ModelDownloader
    from src.inference import OpenVLAInferenceEngine
    from src.model import OpenVLAModelLoader
    from src.webcam import WebcamRunner

    container = None
    if not skip_model_download:
        print("\nChecking OpenVLA Model files...")
        downloader = ModelDownloader()
        try:
            model_path = downloader.download()
            downloader.verify_integrity(model_path)
        except Exception as exc:
            print(f"\nWarning: Model download failed ({exc}). Continuing with mock inference fallback.")
            errors_logger.error(f"Launcher model download failed: {exc}")

        print("\nLoading OpenVLA Model onto compute backend...")
        loader = OpenVLAModelLoader(system_info=sys_info)
        try:
            container = loader.load_model()
        except Exception as exc:
            print(f"\nWarning: Could not load full 7B model directly ({exc}).")
            print("Falling back to synthetic/mock execution for self-test & webcam.")

    engine = OpenVLAInferenceEngine(model_container=container)
    print("\nRunning Self Test...")
    self_test_passed = engine.run_self_test()
    if not self_test_passed:
        print("\nWarning: Self test failed. Proceeding with caution.")

    print("\nInitializing Live Webcam Inference Engine...")
    runner = WebcamRunner(inference_engine=engine)
    runner.run_live_loop()

    print("\n" + "=" * 50)
    print("     OpenVLA Auto execution session completed.     ")
    print("=" * 50 + "\n")
    return 0


def _run_simulation_mode(headless: bool, episodes: int, seed: int) -> int:
    from src.sim3d_env import Drone3DSimEnv
    from src.tsa_rules import create_default_hs_2026_profile

    profile = create_default_hs_2026_profile()
    env = Drone3DSimEnv(profile=profile, seed=seed)

    print(f"Running TSA simulation mode for {episodes} episode(s). Headless={headless}")
    total_score = 0.0
    for episode_index in range(episodes):
        observation = env.reset(seed=seed + episode_index, episode_index=episode_index)
        done = False
        episode_reward = 0.0
        step_count = 0
        while not done:
            action = env.expert_action()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            if not headless and step_count % 2 == 0:
                frame = env.render()
                import cv2

                cv2.imshow("OpenVLA Auto - TSA Sim", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    done = True
                    break
        total_score += env.last_metrics.score if env.last_metrics else episode_reward
        print(f"Episode {episode_index + 1}: score={env.last_metrics.score:.2f}, gates={env.last_metrics.gates_cleared}, collisions={env.last_metrics.collisions}")

    if not headless:
        import cv2

        cv2.destroyAllWindows()

    print(f"Average score: {total_score / max(1, episodes):.2f}")
    return 0


def _run_train_mode(episodes: int, epochs: int, seed: int, checkpoint: str | None) -> int:
    from src.train import train_policy

    summary = train_policy(episodes=episodes, epochs=epochs, seed=seed, checkpoint_path=Path(checkpoint) if checkpoint else None)
    print(summary.to_terminal_text())
    return 0


def _run_eval_mode(episodes: int, seed: int, checkpoint: str | None, report: str | None, headless: bool) -> int:
    from src.eval import evaluate_policy

    result = evaluate_policy(
        episodes=episodes,
        seed=seed,
        checkpoint_path=Path(checkpoint) if checkpoint else None,
        report_path=Path(report) if report else None,
        headless=headless,
    )
    print(result.to_terminal_text())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.mode == "live":
            return _run_live_mode(skip_install=args.no_install, skip_model_download=args.no_model_download)
        if args.mode == "simulate":
            return _run_simulation_mode(headless=args.headless, episodes=args.episodes or 2, seed=args.seed)
        if args.mode == "train":
            return _run_train_mode(episodes=args.episodes or 8, epochs=args.epochs or 4, seed=args.seed, checkpoint=args.checkpoint)
        if args.mode == "eval":
            return _run_eval_mode(episodes=args.episodes or 5, seed=args.seed, checkpoint=args.checkpoint, report=args.report, headless=args.headless)
        parser.error(f"Unsupported mode: {args.mode}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        return 0
    except Exception as exc:
        errors_logger.error(f"Unhandled exception in launcher main: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
