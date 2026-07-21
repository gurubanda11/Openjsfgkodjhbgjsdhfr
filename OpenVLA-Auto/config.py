"""
Configuration settings for OpenVLA Auto.
Provides path constants, dependency lists, PyTorch wheel index URLs, and model settings.
"""

from pathlib import Path
import sys

# Directory Structure
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = MODELS_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"
VENV_DIR = BASE_DIR / ".venv"

# Ensure essential directories exist
for directory in [MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hugging Face Model Configuration
DEFAULT_MODEL_ID = "openvla/openvla-7b"
TOKEN_FILE = CACHE_DIR / ".hf_token"

# Log Files
INSTALL_LOG = LOGS_DIR / "install.log"
MODEL_LOG = LOGS_DIR / "model.log"
WEBCAM_LOG = LOGS_DIR / "webcam.log"
ERRORS_LOG = LOGS_DIR / "errors.log"

# PyTorch Wheel Index URLs
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Core Required Packages
CORE_REQUIREMENTS = [
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "accelerate",
    "opencv-python",
    "pillow",
    "numpy",
    "tqdm",
    "requests",
    "pyyaml",
    "psutil",
    "py-cpuinfo",
]
