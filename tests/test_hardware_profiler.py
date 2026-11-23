"""Unit tests for Hardware-Aware Local AI Profiler (Phase F of EDXSO Milestone).

Covers:
- RAM calculation (total, available, used in GB)
- CPU information gathering (cores, architecture)
- NVIDIA GPU detection parsing
- Hardware tier categorization (high_performance, workstation, standard_laptop)
- Model recommendations per tier
- Advisories and warnings
"""

from unittest.mock import patch
import pytest

from sellbird.hardware_profiler import get_hardware_profile, _get_ram_info, _get_cpu_info, _get_nvidia_gpu_info


def test_hardware_profile_basic():
    profile = get_hardware_profile()
    assert "tier" in profile
    assert profile["tier"] in ("high_performance", "workstation", "standard_laptop")
    assert "ram" in profile
    assert profile["ram"]["total_gb"] > 0
    assert "cpu" in profile
    assert profile["cpu"]["logical_cores"] >= 1
    assert "recommended_models" in profile
    assert "qwen3:4b-instruct-2507-q4_K_M" in profile["recommended_models"]
    assert profile["default_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert profile["inference_provider"] == "ollama"


def test_hardware_tier_high_performance():
    with patch("sellbird.hardware_profiler._get_ram_info", return_value={"total_gb": 32.0, "available_gb": 20.0, "used_gb": 12.0, "percent_used": 37.5}), \
         patch("sellbird.hardware_profiler._get_nvidia_gpu_info", return_value={"has_nvidia_gpu": True, "gpu_name": "RTX 4090", "vram_total_gb": 16.0, "vram_free_gb": 14.0, "driver_version": "550.0"}):
        profile = get_hardware_profile()
        assert profile["tier"] == "high_performance"
        assert "llama3.1:8b" in profile["recommended_models"]


def test_hardware_tier_standard_laptop_with_warnings():
    with patch("sellbird.hardware_profiler._get_ram_info", return_value={"total_gb": 6.0, "available_gb": 1.5, "used_gb": 4.5, "percent_used": 75.0}), \
         patch("sellbird.hardware_profiler._get_nvidia_gpu_info", return_value={"has_nvidia_gpu": False, "gpu_name": None, "vram_total_gb": 0.0, "vram_free_gb": 0.0, "driver_version": None}):
        profile = get_hardware_profile()
        assert profile["tier"] == "standard_laptop"
        assert any("< 8 GB" in w for w in profile["warnings"])
        assert any("CPU-only" in w for w in profile["warnings"])
