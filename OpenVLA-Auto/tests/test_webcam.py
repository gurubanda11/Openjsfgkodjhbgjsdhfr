"""
Unit tests for webcam runner and HUD overlay.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.webcam import WebcamRunner
from src.inference import InferenceResult


class TestWebcamRunner(unittest.TestCase):

    def test_detect_webcams(self):
        cams = WebcamRunner.detect_webcams(max_tested=2)
        self.assertIsInstance(cams, list)

    def test_hud_overlay_rendering(self):
        runner = WebcamRunner()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_result = InferenceResult(
            action_vector=[0.01, -0.02, 0.05, 0.0, 0.0, 0.01, 1.0],
            action_text="Action: [0.01, -0.02, 0.05, 0.00, 0.00, 0.01, 1.00]",
            latency_ms=42.0,
            device="CUDA",
        )
        overlay = runner.render_hud_overlay(blank_frame, dummy_result, fps=30.0)

        self.assertEqual(overlay.shape, (480, 640, 3))
        # Ensure HUD drew pixels on top bar
        self.assertFalse(np.array_equal(overlay[:50, :50], blank_frame[:50, :50]))


if __name__ == "__main__":
    unittest.main()
