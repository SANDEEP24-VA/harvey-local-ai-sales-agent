"""Hardware-aware local AI profiler.

Detects host CPU, RAM, and NVIDIA GPU/VRAM to categorize the system into
hardware tiers (standard_laptop, workstation, high_performance) and provide
local LLM recommendations, memory warnings, and operational guidance.
"""

import logging
import os
import platform
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _get_ram_info() -> dict[str, float]:
    """Retrieve system RAM in gigabytes (cross-platform, zero external dependencies required)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "percent_used": round(mem.percent, 1),
        }
    except ImportError:
        pass

    # Windows fallback via ctypes
    if platform.system() == "Windows":
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
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = stat.ullTotalPhys / (1024 ** 3)
                avail = stat.ullAvailPhys / (1024 ** 3)
                used = total - avail
                pct = (used / total) * 100.0 if total > 0 else 0.0
                return {
                    "total_gb": round(total, 2),
                    "available_gb": round(avail, 2),
                    "used_gb": round(used, 2),
                    "percent_used": round(pct, 1),
                }
        except Exception:
            pass

    # Generic fallback
    return {
        "total_gb": 8.0,
        "available_gb": 4.0,
        "used_gb": 4.0,
        "percent_used": 50.0,
    }


def _get_cpu_info() -> dict[str, Any]:
    """Retrieve CPU architecture, core counts, and processor name."""
    logical_cores = os.cpu_count() or 4
    physical_cores = logical_cores
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
    except ImportError:
        pass

    processor = platform.processor() or platform.machine()
    return {
        "architecture": platform.machine(),
        "processor": processor,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "system": platform.system(),
    }


def _get_nvidia_gpu_info() -> dict[str, Any]:
    """Safely query nvidia-smi for GPU name, total VRAM, free VRAM, and driver."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "has_nvidia_gpu": False,
            "gpu_name": None,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver_version": None,
        }

    try:
        cmd = [
            nvidia_smi,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                total_mb = float(parts[1])
                free_mb = float(parts[2])
                driver = parts[3]
                return {
                    "has_nvidia_gpu": True,
                    "gpu_name": name,
                    "vram_total_gb": round(total_mb / 1024, 2),
                    "vram_free_gb": round(free_mb / 1024, 2),
                    "driver_version": driver,
                }
    except Exception as e:
        logger.debug(f"nvidia-smi query skipped or failed: {e}")

    return {
        "has_nvidia_gpu": False,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "driver_version": None,
    }


def get_hardware_profile() -> dict[str, Any]:
    """Compute the hardware profile, tier rating, and Ollama model recommendations."""
    ram = _get_ram_info()
    cpu = _get_cpu_info()
    gpu = _get_nvidia_gpu_info()

    total_ram = ram["total_gb"]
    vram = gpu["vram_total_gb"]

    # Tier classification
    if (gpu["has_nvidia_gpu"] and vram >= 8.0) or total_ram >= 32.0:
        tier = "high_performance"
        tier_label = "High Performance (Dedicated GPU / Workstation RAM)"
        recommended_models = [
            "qwen3:4b-instruct-2507-q4_K_M",
            "qwen2.5:7b",
            "llama3.1:8b",
        ]
        tier_description = "High capacity. Capable of hosting 7B-8B parameter models with fast full-context inference."
    elif (gpu["has_nvidia_gpu"] and vram >= 4.0) or total_ram >= 16.0:
        tier = "workstation"
        tier_label = "Workstation (Balanced CPU / Mid-tier GPU)"
        recommended_models = [
            "qwen3:4b-instruct-2507-q4_K_M",
            "qwen2.5:3b",
            "phi3:mini",
        ]
        tier_description = "Balanced local inference. Ideal for 3B-4B parameter models with excellent response latency."
    else:
        tier = "standard_laptop"
        tier_label = "Standard Laptop (Resource-Conscious)"
        recommended_models = [
            "qwen3:4b-instruct-2507-q4_K_M",
            "qwen2.5:1.5b",
            "smollm2:1.7b",
        ]
        tier_description = "CPU-focused or integrated graphics. Quantized 4B or compact 1.5B-3B models run efficiently."

    warnings = []
    if total_ram < 8.0:
        warnings.append(
            f"WARNING: Total RAM is {total_ram} GB (< 8 GB recommended). Local Ollama may face high latency or OOM conditions."
        )
    if ram["available_gb"] < 2.5:
        warnings.append(
            f"WARNING: Only {ram['available_gb']} GB RAM available. Close unused background applications for best performance."
        )
    if not gpu["has_nvidia_gpu"]:
        warnings.append(
            "NOTE: No dedicated NVIDIA GPU detected. Local Ollama will run in CPU-only mode."
        )

    return {
        "tier": tier,
        "tier_label": tier_label,
        "tier_description": tier_description,
        "ram": ram,
        "cpu": cpu,
        "gpu": gpu,
        "recommended_models": recommended_models,
        "default_model": "qwen3:4b-instruct-2507-q4_K_M",
        "warnings": warnings,
        "inference_provider": "ollama",
    }
