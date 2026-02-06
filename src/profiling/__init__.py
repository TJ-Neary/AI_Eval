"""
Hardware Profiling Module

Detects system hardware and monitors performance during benchmarks.

Usage:
    from src.profiling import detect_hardware, get_memory_usage

    profile = detect_hardware()
    print(f"Running on {profile.chip_name} with {profile.ram_gb}GB RAM")
"""

from .hardware import (
    detect_hardware,
    get_memory_usage,
    get_cpu_usage,
    HardwareProfile,
    ChipType,
    AppleSiliconTier,
)

__all__ = [
    "detect_hardware",
    "get_memory_usage",
    "get_cpu_usage",
    "HardwareProfile",
    "ChipType",
    "AppleSiliconTier",
]
