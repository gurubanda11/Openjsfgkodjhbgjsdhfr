"""
Virtual Environment Manager and Dependency Installer for OpenVLA Auto.
Handles automatic venv creation, activation re-exec, PyTorch wheel resolution,
pip package installation with progress tracking, speed/ETA reporting, and retry logic.
"""

import sys
import subprocess
import time
import venv
from pathlib import Path
from typing import List, Optional, Tuple

from config import VENV_DIR, CORE_REQUIREMENTS, PYTORCH_CUDA_INDEX, PYTORCH_CPU_INDEX
from src.device import detect_system_info, SystemInfo
from src.utils import install_logger, errors_logger


class VenvManager:
    """
    Manages creation and auto-activation/re-execution of virtual environment.
    """

    @staticmethod
    def get_venv_python() -> Path:
        if sys.platform == "win32":
            return VENV_DIR / "Scripts" / "python.exe"
        return VENV_DIR / "bin" / "python"

    @classmethod
    def is_in_venv(cls) -> bool:
        return sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")

    @classmethod
    def ensure_venv(cls) -> None:
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
        if not cls.is_in_venv():
            cls.ensure_venv()
            venv_python = cls.get_venv_python()
            if not venv_python.exists():
                raise FileNotFoundError(f"Virtualenv python executable not found at {venv_python}")

            install_logger.info(f"Re-executing process inside virtual environment: {venv_python}")
            args = [str(venv_python)] + sys.argv
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

    def _distribution_name_to_import_name(self, pkg: str) -> str:
        pkg_name = pkg.split("==")[0].split(">=")[0].split("<")[0].strip().replace("-", "_")
        if pkg_name == "opencv_python":
            return "cv2"
        if pkg_name == "py_cpuinfo":
            return "cpuinfo"
        if pkg_name == "pillow":
            return "PIL"
        if pkg_name == "pyyaml":
            return "yaml"
        return pkg_name

    def get_missing_packages(self) -> List[str]:
        missing: List[str] = []
        for pkg in CORE_REQUIREMENTS:
            import_name = self._distribution_name_to_import_name(pkg)
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pkg)
        return missing

    def _run_pip_command(self, args: List[str], max_retries: int = 3) -> bool:
        for attempt in range(1, max_retries + 1):
            install_logger.info(f"Running pip command (attempt {attempt}/{max_retries}): {' '.join(args)}")
            start_time = time.time()
            try:
                result = subprocess.run(args, capture_output=True, text=True)
                elapsed = time.time() - start_time
                if result.returncode == 0:
                    install_logger.info(f"Command succeeded in {elapsed:.1f}s")
                    return True

                err_out = (result.stderr or result.stdout or "Unknown pip error").strip()
                install_logger.warning(f"Command failed in {elapsed:.1f}s: {err_out[-1000:]}")
                errors_logger.error(err_out)
            except Exception as e:
                install_logger.warning(f"pip execution error: {e}")
                errors_logger.error(f"pip execution error: {e}")

            if attempt < max_retries:
                wait_sec = attempt * 2
                install_logger.info(f"Retrying in {wait_sec}s...")
                time.sleep(wait_sec)

        return False

    def install_packages(self, packages: List[str], extra_index_url: Optional[str] = None, max_retries: int = 3) -> bool:
        if not packages:
            install_logger.info("All required packages are already installed.")
            return True

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if extra_index_url:
            cmd.extend(["--extra-index-url", extra_index_url])
        cmd.extend(packages)

        install_logger.info(f"Installing packages: {', '.join(packages)}")
        return self._run_pip_command(cmd, max_retries=max_retries)

    def _install_pytorch_stack(self) -> bool:
        backend = self.sys_info.backend
        if backend == "CUDA":
            torch_pkgs = ["torch>=2.3.1", "torchvision>=0.18.1"]
            return self.install_packages(torch_pkgs, extra_index_url=PYTORCH_CUDA_INDEX, max_retries=3)

        if backend == "MPS":
            torch_pkgs = ["torch>=2.3.1", "torchvision>=0.18.1"]
            return self.install_packages(torch_pkgs, extra_index_url=None, max_retries=3)

        torch_pkgs = ["torch>=2.3.1", "torchvision>=0.18.1"]
        return self.install_packages(torch_pkgs, extra_index_url=PYTORCH_CPU_INDEX, max_retries=3)

    def _validate_critical_imports(self) -> Tuple[bool, str]:
        try:
            from transformers import AutoProcessor  # noqa: F401
        except Exception as e:
            return False, f"Transformers import failed: {e}"

        try:
            from transformers import AutoModelForVision2Seq  # noqa: F401
            return True, "ok"
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText  # noqa: F401
                return True, "fallback-ok"
            except Exception as e:
                return False, f"Missing required transformers auto model class: {e}"

    def _explain_failure_and_suggest_fix(self) -> None:
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
            "   - PyTorch CUDA builds require multiple GB of disk space.",
            "3. Permission Issues:",
            "   - Run launcher with elevated privileges or inside user directory.",
            "4. Manual Override:",
            "   - Activate virtual environment (.venv) manually and run:",
            "     python -m pip install -U -r requirements.txt",
            "=" * 50,
        ]
        text = "\n".join(msg)
        print(text)
        errors_logger.error(text)

    def run_installation(self) -> bool:
        if not self._install_pytorch_stack():
            self._explain_failure_and_suggest_fix()
            return False

        missing = self.get_missing_packages()
        if missing and not self.install_packages(missing, extra_index_url=None, max_retries=3):
            self._explain_failure_and_suggest_fix()
            return False

        ok, detail = self._validate_critical_imports()
        if not ok:
            errors_logger.error(detail)
            repair = [
                "transformers>=4.46.0",
                "huggingface_hub>=0.24.6",
                "safetensors>=0.4.3",
            ]
            if not self.install_packages(repair, extra_index_url=None, max_retries=3):
                self._explain_failure_and_suggest_fix()
                return False
            ok2, detail2 = self._validate_critical_imports()
            if not ok2:
                errors_logger.error(detail2)
                self._explain_failure_and_suggest_fix()
                return False

        install_logger.info("Dependency installation and validation completed successfully.")
        return True


if __name__ == "__main__":
    VenvManager.activate_or_reexec()
    installer = DependencyInstaller()
    success = installer.run_installation()
    if not success:
        sys.exit(1)
