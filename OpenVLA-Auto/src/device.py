"""
Hardware and Operating System detection module for OpenVLA Auto.
Provides SystemInfo dataclass and terminal UI formatting.
"""

from dataclasses import dataclass
import os
import platform
import subprocess
import sys
from typing import Optional


@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str
    cuda_available: bool
    cuda_version: str
    mps_available: bool
    backend: str  # "CUDA", "MPS", or "CPU"

    def format_terminal_ui(self) -> str:
        """
        Renders the exact formatted System Information terminal UI display.
        """
        lines = [
            "=" * 50,
            "              System Information              ",
            "=" * 50,
            "",
            "Operating System",
            f"{self.os_name} {self.os_version}".strip(),
            "",
            "CPU",
            f"{self.cpu_name} ({self.cpu_cores} Cores)",
            "",
            "RAM",
            f"{self.ram_total_gb:.1f} GB ({self.ram_available_gb:.1f} GB Available)",
            "",
            "GPU",
            self.gpu_name,
            "",
            "CUDA",
            self.cuda_version if self.cuda_available else "Not Available",
            "",
            "Backend",
            self.backend,
            "=" * 50,
        ]
        return "\n".join(lines)


def get_cpu_info() -> str:
    """
    Detects CPU brand/model name robustly across Windows, macOS, and Linux.
    """
    # 1. Try py-cpuinfo if installed
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if "brand_raw" in info and info["brand_raw"]:
            return info["brand_raw"]
    except ImportError:
        pass
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Windows":
            cmd = "wmic cpu get name"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if len(lines) > 1:
                return lines[1]
        elif system == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            output = subprocess.check_output(cmd).decode().strip()
            if output:
                return output
        elif system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass

    processor = platform.processor()
    if processor:
        return processor
    return platform.machine() or "Generic CPU"


def get_ram_info() -> tuple[float, float]:
    """
    Returns total RAM (GB) and available RAM (GB).
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        pass
    except Exception:
        pass

    system = platform.system()
    if system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3), stat.ullAvailPhys / (1024**3)
        except Exception:
            pass
    elif system == "Linux":
        try:
            mem_total = 0.0
            mem_available = 0.0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = float(line.split()[1]) * 1024 / (1024**3)
                    elif line.startswith("MemAvailable:"):
                        mem_available = float(line.split()[1]) * 1024 / (1024**3)
            if mem_total > 0:
                return mem_total, mem_available
        except Exception:
            pass

    return 8.0, 4.0  # Fallback default estimate if psutil is not available


def detect_system_info() -> SystemInfo:
    """
    Performs full hardware and operating system inspection.
    """
    system = platform.system()
    if system == "Windows":
        os_name = "Windows"
        os_ver = platform.release()
        if os_ver == "10":
            # Distinguish Windows 11 by build number
            try:
                build = int(platform.version().split(".")[-1])
                if build >= 22000:
                    os_ver = "11"
            except Exception:
                pass
    elif system == "Darwin":
        os_name = "macOS"
        os_ver = platform.mac_ver()[0] or platform.release()
    elif system == "Linux":
        os_name = "Linux"
        try:
            with open("/etc/os-release", "r") as f:
                d = dict(
                    line.strip().split("=", 1) for line in f if "=" in line
                )
                os_name = d.get("NAME", "Linux").strip('"')
                os_ver = d.get("VERSION_ID", platform.release()).strip('"')
        except Exception:
            os_ver = platform.release()
    else:
        os_name = system
        os_ver = platform.release()

    cpu_name = get_cpu_info()
    cpu_cores = os.cpu_count() or 4
    ram_total, ram_avail = get_ram_info()

    # GPU & CUDA & MPS Detection
    gpu_name = "Integrated / Software Graphics"
    cuda_available = False
    cuda_version = "Not Available"
    mps_available = False

    # Check via torch first if available
    torch_loaded = False
    try:
        import torch

        torch_loaded = True
        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.version.cuda or "Available"
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_available = True
            gpu_name = "Apple Silicon GPU (MPS)"
    except ImportError:
        pass
    except Exception:
        pass

    # If torch is not installed or didn't report CUDA, check via CLI / system tools
    if not cuda_available:
        try:
            output = subprocess.check_output(
                "nvidia-smi --query-gpu=gpu_name --format=csv,noheader",
                shell=True,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if output:
                gpu_name = output.splitlines()[0]
                cuda_available = True
                # Get CUDA driver version
                try:
                    smi_out = subprocess.check_output(
                        "nvidia-smi", shell=True, stderr=subprocess.DEVNULL
                    ).decode()
                    for line in smi_out.splitlines():
                        if "CUDA Version:" in line:
                            cuda_version = line.split("CUDA Version:")[1].split()[0].strip()
                            break
                except Exception:
                    cuda_version = "12.x"
        except Exception:
            pass

    # Check Apple Silicon MPS if on macOS and not set yet
    if system == "Darwin" and not mps_available:
        machine = platform.machine()
        if "arm" in machine.lower() or "aarch64" in machine.lower():
            mps_available = True
            gpu_name = "Apple Silicon GPU (MPS)"

    # Determine recommended backend
    if cuda_available:
        backend = "CUDA"
    elif mps_available:
        backend = "MPS"
    else:
        backend = "CPU"

    return SystemInfo(
        os_name=os_name,
        os_version=os_ver,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        mps_available=mps_available,
        backend=backend,
    )


if __name__ == "__main__":
    sys_info = detect_system_info()
    print(sys_info.format_terminal_ui())
