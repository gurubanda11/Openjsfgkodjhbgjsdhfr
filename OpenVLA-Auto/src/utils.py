"""
Utility functions for OpenVLA Auto including logging, token management, and file operations.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional
import sys

from config import LOGS_DIR, TOKEN_FILE, INSTALL_LOG, MODEL_LOG, WEBCAM_LOG, ERRORS_LOG


def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to both a file and standard output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger is re-initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Errors File Handler for ERROR and CRITICAL logs
    error_handler = logging.FileHandler(ERRORS_LOG, encoding="utf-8")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    # Stream Handler (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


# Pre-configured loggers for components
install_logger = setup_logger("Installer", INSTALL_LOG)
model_logger = setup_logger("ModelLoader", MODEL_LOG)
webcam_logger = setup_logger("Webcam", WEBCAM_LOG)
errors_logger = setup_logger("Errors", ERRORS_LOG)


def format_bytes(size_bytes: int | float) -> str:
    """
    Converts bytes into a human-readable string (KB, MB, GB, etc.).
    """
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(size_bytes)
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def compute_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Computes the SHA256 hash of a given file.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_hf_token() -> Optional[str]:
    """
    Retrieves stored Hugging Face token if available.
    """
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            return token if token else None
        except Exception as e:
            errors_logger.error(f"Failed to read Hugging Face token: {e}")
    return None


def save_hf_token(token: str) -> None:
    """
    Saves a Hugging Face token securely.
    """
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token.strip(), encoding="utf-8")
        install_logger.info("Hugging Face token saved successfully.")
    except Exception as e:
        errors_logger.error(f"Failed to save Hugging Face token: {e}")
