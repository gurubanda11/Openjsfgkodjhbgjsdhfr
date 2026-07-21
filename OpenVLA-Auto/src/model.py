"""
Model loader module for OpenVLA.
Handles precision selection, device placement, progress display, and memory profiling.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional, Tuple

import torch
from transformers import AutoProcessor

try:
    # Available on newer transformers versions used by OpenVLA stacks.
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:  # pragma: no cover - fallback path for older/newer API shifts
    # Robust fallback for environments where AutoModelForVision2Seq is unavailable.
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq  # type: ignore

from config import DEFAULT_MODEL_ID, CACHE_DIR
from src.device import detect_system_info, SystemInfo
from src.utils import model_logger, errors_logger, format_bytes


@dataclass
class LoadedModelContainer:
    model: Any
    processor: Any
    device: str
    dtype: torch.dtype
    param_count: int
    memory_used_bytes: int


class OpenVLAModelLoader:
    """
    Loads OpenVLA Vision-Language-Action model and processor onto optimal compute device.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Path = CACHE_DIR,
        system_info: Optional[SystemInfo] = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.sys_info = system_info or detect_system_info()

    def select_device_and_dtype(self) -> Tuple[str, torch.dtype]:
        """
        Selects optimal device string and torch dtype based on system capabilities.
        """
        backend = self.sys_info.backend

        if backend == "CUDA" and torch.cuda.is_available():
            device = "cuda"
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif backend == "MPS" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        return device, dtype

    def calculate_memory_usage(self, device: str) -> int:
        """
        Calculates memory usage in bytes for the active device backend.
        """
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        elif device == "mps":
            return 0
        else:
            try:
                import psutil

                process = psutil.Process()
                return process.memory_info().rss
            except Exception:
                return 0

    def load_model(self) -> LoadedModelContainer:
        """
        Loads the model and processor, returning a LoadedModelContainer.
        """
        device_str, dtype = self.select_device_and_dtype()
        model_logger.info(f"Loading OpenVLA model '{self.model_id}'...")
        model_logger.info(f"Target Device: {device_str.upper()} | Precision: {dtype}")

        start_time = time.time()

        try:
            model_logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )

            model_logger.info("Loading model weights (this may take a few moments)...")

            load_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "cache_dir": str(self.cache_dir),
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            model = AutoModelForVision2Seq.from_pretrained(**load_kwargs)

            model_logger.info(f"Moving model to device ({device_str.upper()})...")
            model.to(device_str)
            model.eval()

            elapsed = time.time() - start_time
            param_count = sum(p.numel() for p in model.parameters())
            mem_used = self.calculate_memory_usage(device_str)

            model_logger.info(f"Model loaded successfully in {elapsed:.2f}s!")
            model_logger.info(f"Total Parameters: {param_count:,}")
            model_logger.info(f"Estimated Memory Footprint: {format_bytes(mem_used)}")

            return LoadedModelContainer(
                model=model,
                processor=processor,
                device=device_str,
                dtype=dtype,
                param_count=param_count,
                memory_used_bytes=mem_used,
            )

        except Exception as e:
            err_msg = f"Failed to load OpenVLA model '{self.model_id}': {e}"
            model_logger.error(err_msg)
            errors_logger.error(err_msg)
            raise RuntimeError(err_msg) from e


if __name__ == "__main__":
    loader = OpenVLAModelLoader()
    container = loader.load_model()
    print(f"Loaded successfully on {container.device} with {container.param_count} parameters.")
