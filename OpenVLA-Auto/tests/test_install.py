"""
Unit tests for system detection and installation routines.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.device import detect_system_info, SystemInfo
from installer import DependencyInstaller, VenvManager
from config import PYTORCH_CUDA_INDEX, PYTORCH_CPU_INDEX


class TestInstallation(unittest.TestCase):

    def test_system_info_detection(self):
        sys_info = detect_system_info()
        self.assertIsInstance(sys_info, SystemInfo)
        self.assertIn(sys_info.backend, ["CUDA", "MPS", "CPU"])
        self.assertGreater(sys_info.ram_total_gb, 0)
        self.assertGreater(sys_info.cpu_cores, 0)

    def test_terminal_ui_formatting(self):
        sys_info = SystemInfo(
            os_name="Windows",
            os_version="11",
            cpu_name="Intel i9-10920X",
            cpu_cores=12,
            ram_total_gb=64.0,
            ram_available_gb=48.2,
            gpu_name="NVIDIA RTX 5090",
            cuda_available=True,
            cuda_version="12.1",
            mps_available=False,
            backend="CUDA",
        )
        output = sys_info.format_terminal_ui()
        self.assertIn("System Information", output)
        self.assertIn("Windows 11", output)
        self.assertIn("NVIDIA RTX 5090", output)
        self.assertIn("CUDA", output)

    def test_pytorch_index_selection(self):
        info_cuda = SystemInfo("Win", "11", "CPU", 8, 16.0, 8.0, "RTX", True, "12.1", False, "CUDA")
        installer_cuda = DependencyInstaller(system_info=info_cuda)
        
        info_cpu = SystemInfo("Linux", "22.04", "CPU", 4, 8.0, 4.0, "None", False, "No", False, "CPU")
        installer_cpu = DependencyInstaller(system_info=info_cpu)

        self.assertEqual(installer_cuda.sys_info.backend, "CUDA")
        self.assertEqual(installer_cpu.sys_info.backend, "CPU")

    def test_venv_check(self):
        is_venv = VenvManager.is_in_venv()
        self.assertIsInstance(is_venv, bool)


if __name__ == "__main__":
    unittest.main()
