"""
Unit tests for model loading and inference engine.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.device import SystemInfo
from src.model import OpenVLAModelLoader
from src.inference import OpenVLAInferenceEngine, InferenceResult


class TestModelEngine(unittest.TestCase):

    def test_device_dtype_selection(self):
        info_cuda = SystemInfo("Win", "11", "CPU", 8, 16.0, 8.0, "RTX", True, "12.1", False, "CUDA")
        loader_cuda = OpenVLAModelLoader(system_info=info_cuda)
        device, dtype = loader_cuda.select_device_and_dtype()
        self.assertIn(device, ["cuda", "cpu"])

        info_cpu = SystemInfo("Linux", "22.04", "CPU", 4, 8.0, 4.0, "None", False, "No", False, "CPU")
        loader_cpu = OpenVLAModelLoader(system_info=info_cpu)
        device_cpu, dtype_cpu = loader_cpu.select_device_and_dtype()
        self.assertEqual(device_cpu, "cpu")

    def test_inference_engine_prediction(self):
        engine = OpenVLAInferenceEngine()
        test_img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        result = engine.run_inference(test_img)

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(len(result.action_vector), 7)
        self.assertGreater(result.latency_ms, 0)
        self.assertTrue(result.action_text.startswith("Action:"))

    def test_numpy_image_inference(self):
        engine = OpenVLAInferenceEngine()
        test_np = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.run_inference(test_np)
        self.assertEqual(len(result.action_vector), 7)

    def test_self_test(self):
        engine = OpenVLAInferenceEngine()
        success = engine.run_self_test()
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
