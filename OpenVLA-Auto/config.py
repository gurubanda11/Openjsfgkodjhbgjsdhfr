"""
Configuration settings for OpenVLA Auto.
Provides path constants, dependency lists, PyTorch wheel index URLs, and model settings.
"""

from pathlib import Path

# Directory Structure
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = MODELS_DIR / "cache"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SIM_OUTPUT_DIR = ARTIFACTS_DIR / "sim"
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"
VENV_DIR = BASE_DIR / ".venv"

# Ensure essential directories exist
for directory in [MODELS_DIR, CACHE_DIR, ARTIFACTS_DIR, CHECKPOINTS_DIR, REPORTS_DIR, SIM_OUTPUT_DIR, LOGS_DIR]:
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
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "transformers>=4.46.0",
    "huggingface_hub>=0.24.6",
    "safetensors>=0.4.3",
    "accelerate>=0.33.0",
    "opencv-python>=4.10.0.84",
    "pillow>=10.4.0",
    "numpy>=1.26.4",
    "tqdm>=4.66.5",
    "requests>=2.32.3",
    "pyyaml>=6.0.2",
    "psutil>=6.0.0",
    "py-cpuinfo>=9.0.0",
]

# Live inference settings
DEFAULT_PROMPT = "In: What action should the robot take next based on this scene?\nOut:"
INFERENCE_EVERY_N_FRAMES = 3
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
SIM_PANEL_WIDTH = 480
HUD_HEIGHT = 170

# TSA / simulation defaults
DEFAULT_LAUNCHER_MODE = "live"
DEFAULT_SEED = 2026
DEFAULT_TRAIN_EPISODES = 8
DEFAULT_TRAIN_EPOCHS = 4
DEFAULT_EVAL_EPISODES = 5
DEFAULT_SIM_EPISODES = 2
DEFAULT_CHECKPOINT_NAME = "tsa_policy.pt"
DEFAULT_REPORT_NAME = "tsa_eval_report.json"
