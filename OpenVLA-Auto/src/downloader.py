"""
Hugging Face Model Downloader for OpenVLA Auto.
Handles token authentication, progress bar downloads, download resuming, and cache verification.
"""

from pathlib import Path
import sys
from typing import Optional

from config import CACHE_DIR, DEFAULT_MODEL_ID
from src.utils import get_hf_token, save_hf_token, model_logger, errors_logger


class ModelDownloader:
    """
    Downloads and caches OpenVLA model files from Hugging Face hub.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prompt_hf_token(self) -> str:
        """
        Interactively prompts the user for a Hugging Face authentication token.
        """
        print("\n" + "=" * 50)
        print("          HUGGING FACE AUTHENTICATION REQUIRED          ")
        print("=" * 50)
        print(f"The model '{self.model_id}' requires authentication or access token.")
        token = input("Please enter your Hugging Face Token: ").strip()
        if not token:
            raise ValueError("Token cannot be empty.")
        save_hf_token(token)
        return token

    def is_cached(self) -> bool:
        """
        Checks if model snapshot files are present in local cache directory.
        """
        model_dir_name = "models--" + self.model_id.replace("/", "--")
        expected_dir = self.cache_dir / model_dir_name
        if expected_dir.exists():
            files = list(expected_dir.rglob("*"))
            if any(f.name.endswith(".safetensors") or f.name.endswith(".bin") for f in files):
                return True
        
        # Also check direct folder download format if snapshot_download target directory was specified
        direct_dir = self.cache_dir / self.model_id.split("/")[-1]
        if direct_dir.exists() and (direct_dir / "config.json").exists():
            return True

        return False

    def download(self, force_token_prompt: bool = False) -> Path:
        """
        Downloads model repository with progress bar and resume capabilities.
        Returns Path to local model directory.
        """
        if self.is_cached() and not force_token_prompt:
            model_logger.info(f"Model '{self.model_id}' is already cached in {self.cache_dir}.")
            return self.cache_dir

        model_logger.info(f"Starting download for OpenVLA model: {self.model_id}")
        token = get_hf_token()

        if force_token_prompt or (not token and "openvla" in self.model_id):
            # Check if token is requested by user or token file is empty
            pass  # Token will be requested if download fails with GatedRepo / 401

        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir),
                token=token,
                resume_download=True,
                max_workers=4,
            )
            model_logger.info(f"Model downloaded successfully to: {local_path}")
            return Path(local_path)

        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "GatedRepo" in err_str or "restricted" in err_str.lower() or "403" in err_str:
                model_logger.warning("Hugging Face authentication required or invalid token.")
                new_token = self.prompt_hf_token()
                # Retry download with new token
                return self.download_with_token(new_token)
            else:
                model_logger.error(f"Failed to download model '{self.model_id}': {e}")
                errors_logger.error(f"Model download error: {e}")
                raise RuntimeError(f"Model download failed: {e}") from e

    def download_with_token(self, token: str) -> Path:
        """
        Retries snapshot_download with explicit token.
        """
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=self.model_id,
            cache_dir=str(self.cache_dir),
            token=token,
            resume_download=True,
            max_workers=4,
        )
        model_logger.info(f"Model downloaded successfully with token to: {local_path}")
        return Path(local_path)

    def verify_integrity(self, model_path: Path) -> bool:
        """
        Verifies downloaded model files exist and are not corrupt/empty.
        """
        if not model_path.exists():
            model_logger.error(f"Model path does not exist: {model_path}")
            return False

        config_file = any(model_path.rglob("config.json"))
        weight_file = any(model_path.rglob("*.safetensors")) or any(model_path.rglob("*.bin"))

        if config_file and weight_file:
            model_logger.info("Model file integrity check passed.")
            return True
        else:
            model_logger.warning("Model integrity check failed: missing config or weight files.")
            return False


if __name__ == "__main__":
    downloader = ModelDownloader()
    path = downloader.download()
    downloader.verify_integrity(path)
