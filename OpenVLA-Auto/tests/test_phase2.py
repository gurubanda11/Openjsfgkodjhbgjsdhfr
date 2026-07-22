"""
Smoke tests for the TSA simulation / training / evaluation pipeline.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher import main as launcher_main
from src.eval import evaluate_policy
from src.policy import DronePolicyNet, build_checkpoint_metadata, load_checkpoint, save_checkpoint
from src.sim3d_env import Drone3DSimEnv
from src.train import train_policy
from src.tsa_rules import create_default_hs_2026_profile


class TestPhase2Pipeline(unittest.TestCase):
    def test_launcher_simulate_mode(self) -> None:
        exit_code = launcher_main(["simulate", "--headless", "--episodes", "1", "--no-install", "--seed", "2026"])
        self.assertEqual(exit_code, 0)

    def test_sim_step_loop(self) -> None:
        env = Drone3DSimEnv(profile=create_default_hs_2026_profile(), seed=2026)
        observation = env.reset(seed=2026, episode_index=0)
        self.assertEqual(observation.shape[0], env.observation_dim)
        next_observation, reward, done, info = env.step(env.expert_action())
        self.assertEqual(next_observation.shape[0], env.observation_dim)
        self.assertIsInstance(reward, float)
        self.assertIn("metrics", info)

    def test_checkpoint_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "policy.pt"
            model = DronePolicyNet(obs_dim=26, action_dim=7)
            metadata = build_checkpoint_metadata(26, 7, "TSA HS Drone Challenge", "2026", 2026, "mixed")
            save_checkpoint(checkpoint_path, model, metadata)
            loaded_model, loaded_metadata, payload = load_checkpoint(checkpoint_path)
            self.assertEqual(loaded_metadata.obs_dim, 26)
            self.assertEqual(loaded_metadata.action_dim, 7)
            self.assertIn("model_state_dict", payload)
            self.assertIsInstance(loaded_model, DronePolicyNet)

    def test_train_and_eval_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "tsa_policy.pt"
            report_path = Path(temp_dir) / "eval_report.json"
            summary = train_policy(episodes=1, epochs=1, seed=2026, checkpoint_path=checkpoint_path)
            self.assertTrue(checkpoint_path.exists())
            self.assertGreater(summary.dataset_size, 0)
            result = evaluate_policy(episodes=1, seed=2026, checkpoint_path=checkpoint_path, report_path=report_path, headless=True)
            self.assertTrue(report_path.exists())
            self.assertGreaterEqual(result.episodes, 1)
            self.assertGreaterEqual(result.mean_score, -1000.0)


if __name__ == "__main__":
    unittest.main()