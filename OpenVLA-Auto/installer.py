"""
Virtual Environment Manager and Dependency Installer for OpenVLA Auto.
Handles automatic venv creation, activation re-exec, PyTorch wheel resolution,
pip package installation with progress tracking, speed/ETA reporting, and retry logic.
"""

import os
import sys
import subprocess
import time
import venv
from pathlib import Path
from typing import List, Optional

from config import VENV_DIR, CORE_REQUIREMENTS, PYTORCH_CUDA_INDEX, PYTORCH_CPU_INDEX
from src.device import detect_system_info, SystemInfo
from src.utils import install_logger, errors_logger


class VenvManager:
    """
    Manages creation and auto-activation/re-execution of virtual environment.
    """

    @staticmethod
    def get_venv_python() -> Path:
        """
        Returns path to python executable inside .venv directory.
        """
        if sys.platform == "win32":
            return VENV_DIR / "Scripts" / "python.exe"
        else:
            return VENV_DIR / "bin" / "python"

    @classmethod
    def is_in_venv(cls) -> bool:
        """
        Checks whether the current process is executing within a virtual environment.
        """
        return sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")

    @classmethod
    def ensure_venv(cls) -> None:
        """
        Ensures virtual environment exists. Creates it if missing.
        """
        if not VENV_DIR.exists():
            install_logger.info(f"Creating virtual environment at {VENV_DIR}...")
            try:
                builder = venv.EnvBuilder(with_pip=True)
                builder.create(VENV_DIR)
                install_logger.info("Virtual environment created successfully.")
            except Exception as e:
                errors_logger.error(f"Failed to create virtual environment: {e}")
                raise RuntimeError(f"Virtual environment creation failed: {e}") from e

    @classmethod
    def activate_or_reexec(cls) -> None:
        """
        If not in venv, ensures venv exists and re-executes current script inside venv.
        """
        if not cls.is_in_venv():
            cls.ensure_venv()
            venv_python = cls.get_venv_python()
            if not venv_python.exists():
                raise FileNotFoundError(f"Virtualenv python executable not found at {venv_python}")
            
            install_logger.info(f"Re-executing process inside virtual environment: {venv_python}")
            args = [str(venv_python)] + sys.argv
            # Replace current process or run subprocess and exit
            try:
                res = subprocess.call(args)
                sys.exit(res)
            except Exception as e:
                errors_logger.error(f"Failed to re-execute in venv: {e}")
                sys.exit(1)


class DependencyInstaller:
    """
    Handles installation of PyTorch and core requirements with retries and progress tracking.
    """

    def __init__(self, system_info: Optional[SystemInfo] = None) -> None:
        self.sys_info = system_info or detect_system_info()

    def get_missing_packages(self) -> List[str]:
        """
        Returns list of missing required packages.
        """
        missing = []
        for pkg in CORE_REQUIREMENTS:
            pkg_name = pkg.split("==")[0].split(">=")[0].replace("-", "_")
            if pkg_name == "opencv_python":
                pkg_name = "cv2"
            elif pkg_name == "py_cpuinfo":
                pkg_name = "cpuinfo"
            elif pkg_name == "pillow":
                pkg_name = "PIL"
            elif pkg_name == "pyyaml":
                pkg_name = "yaml"

            try:
                __import__(pkg_name)
            except ImportError:
                missing.append(pkg)
        return missing

    def install_packages(
        self,
        packages: List[str],
        extra_index_url: Optional[str] = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Installs specified packages using pip with progress bar and retry logic.
        """
        if not packages:
            install_logger.info("All required packages are already installed.")
            return True

        cmd = [sys.executable, "-m", "pip", "install"]
        if extra_index_url:
            cmd.extend(["--extra-index-url", extra_index_url])
        cmd.extend(packages)

        install_logger.info(f"Installing packages: {', '.join(packages)}")
        if extra_index_url:
            install_logger.info(f"Using extra index URL: {extra_index_url}")

        for attempt in range(1, max_retries + 1):
            install_logger.info(f"Installation attempt {attempt}/{max_retries}...")
            start_time = time.time()
            
            try:
                # Use subprocess to pipe pip output with real-time feedback
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                # Monitor pip output for progress/download messages
                stdout_lines = []
                stderr_lines = []

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line_str = line.strip()
                        stdout_lines.append(line_str)
                        if "Downloading" in line_str or "Installing" in line_str or "Successfully" in line_str:
                            install_logger.info(f"  [pip] {line_str}")

                _, stderr_data = process.communicate()
                if stderr_data:
                    stderr_lines = stderr_data.strip().splitlines()

                if process.returncode == 0:
                    elapsed = time.time() - start_time
                    install_logger.info(f"Package installation succeeded in {elapsed:.1f}s.")
                    return True
                else:
                    err_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown pip error."
                    install_logger.warning(f"Attempt {attempt} failed: {err_msg}")
                    errors_logger.error(f"Pip installation failed on attempt {attempt}: {err_msg}")

            except Exception as e:
                install_logger.warning(f"Attempt {attempt} encountered exception: {e}")
                errors_logger.error(f"Installation exception: {e}")

            if attempt < max_retries:
                wait_sec = attempt * 2
                install_logger.info(f"Retrying in {wait_sec} seconds...")
                time.sleep(wait_sec)

        # Installation failed after max retries
        self._explain_failure_and_suggest_fix()
        return False

    def _explain_failure_and_suggest_fix(self) -> None:
        """
        Provides detailed failure diagnostics and suggested remedies.
        """
        msg = [
            "\n" + "=" * 50,
            "          INSTALLATION FAILURE DIAGNOSTICS          ",
            "=" * 50,
            "Package installation failed after multiple retries.",
            "",
            "Possible Causes & Suggested Fixes:",
            "1. Network Connectivity / Firewall:",
            "   - Ensure active internet connection.",
            "   - If using a proxy, set HTTP_PROXY and HTTPS_PROXY environment variables.",
            "2. Insufficient Disk Space:",
            "   - PyTorch CUDA builds require up to 4 GB of disk space.",
            "3. Permission Issues:",
            "   - Run launcher with elevated privileges or inside user directory.",
            "4. Manual Override:",
            "   - Activate virtual environment (.venv) manually and run:",
            "     pip install -r requirements.txt",
            "=" * 50,
        ]
        text = "\n".join(msg)
        print(text)
        errors_logger.error(text)

    def run_installation(self) -> bool:
        """
        Detects backend and executes full installation pipeline.
        """
        missing = self.get_missing_packages()
        if not missing:
            install_logger.info("All dependencies are satisfied.")
            return True

        extra_index: Optional[str] = None
        if self.sys_info.backend == "CUDA":
            extra_index = PYTORCH_CUDA_INDEX
            install_logger.info("Selected PyTorch CUDA wheel build index.")
        elif self.sys_info.backend == "MPS":
            extra_index = None
            install_logger.info("Selected PyTorch Apple MPS default build index.")
        else:
            extra_index = PYTORCH_CPU_INDEX
            install_logger.info("Selected PyTorch CPU wheel build index.")

        return self.install_packages(missing, extra_index_url=extra_index)


if __name__ == "__main__":
    VenvManager.activate_or_reexec()
    installer = DependencyInstaller()
    success = installer.run_installation()
    if not success:
        sys.exit(1)
