"""
Hardware Detection and Profiling

Detects system hardware for benchmark context and model compatibility.
Supports Apple Silicon, NVIDIA GPUs, AMD GPUs, and CPU-only configurations.

Based on research from R-005 in DevPlan:
- Apple Silicon: system_profiler, ioreg, sysctl
- NVIDIA: pynvml or nvidia-smi
- AMD: rocm-smi

Usage:
    from src.profiling import detect_hardware, HardwareProfile

    profile = detect_hardware()
    print(f"Chip: {profile.chip_name}, RAM: {profile.ram_gb}GB")
    print(f"Can run 32B model: {profile.can_run_model_size('32B')}")
"""

import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)


class ChipType(Enum):
    """Hardware accelerator type."""

    APPLE_SILICON = auto()
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    CPU_ONLY = auto()
    UNKNOWN = auto()


class AppleSiliconTier(Enum):
    """Apple Silicon chip tiers for model compatibility."""

    M1 = "m1"
    M1_PRO = "m1_pro"
    M1_MAX = "m1_max"
    M1_ULTRA = "m1_ultra"
    M2 = "m2"
    M2_PRO = "m2_pro"
    M2_MAX = "m2_max"
    M2_ULTRA = "m2_ultra"
    M3 = "m3"
    M3_PRO = "m3_pro"
    M3_MAX = "m3_max"
    M3_ULTRA = "m3_ultra"
    M4 = "m4"
    M4_PRO = "m4_pro"
    M4_MAX = "m4_max"
    M4_ULTRA = "m4_ultra"
    UNKNOWN = "unknown"


@dataclass
class HardwareProfile:
    """Complete hardware profile for a system."""

    # System
    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    platform: str = ""

    # CPU
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0

    # Memory
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU/Accelerator
    chip_type: ChipType = ChipType.UNKNOWN
    chip_name: str = ""
    gpu_cores: int = 0
    gpu_memory_gb: float = 0.0
    neural_engine_cores: int = 0

    # Apple Silicon specific
    apple_tier: AppleSiliconTier = AppleSiliconTier.UNKNOWN
    unified_memory: bool = False
    memory_bandwidth_gbps: float = 0.0

    # NVIDIA specific
    cuda_version: str = ""
    driver_version: str = ""

    # Capabilities
    supports_mps: bool = False  # Apple Metal Performance Shaders
    supports_cuda: bool = False
    supports_mlx: bool = False

    # Extra metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def can_run_model_size(self, size: str) -> bool:
        """
        Check if this hardware can run a model of given size.

        Args:
            size: Model size string like "7B", "13B", "32B", "70B"

        Returns:
            True if likely to run (with quantization), False otherwise.
        """
        # Parse size
        size_match = re.match(r"(\d+)([BbMm])", size)
        if not size_match:
            return True  # Unknown size, assume it works

        num = int(size_match.group(1))
        unit = size_match.group(2).upper()

        if unit == "M":
            params_billions = num / 1000
        else:
            params_billions = num

        # Rule of thumb: Q4 quantization uses ~0.5GB per billion parameters
        # Plus ~2GB overhead for KV cache at reasonable context
        estimated_ram_q4 = (params_billions * 0.5) + 2

        # Check against available RAM (leave 4GB for system)
        usable_ram = self.ram_gb - 4

        if usable_ram >= estimated_ram_q4:
            return True

        return False

    def recommended_quantization(self, model_size: str) -> str:
        """
        Recommend quantization level for a model size.

        Args:
            model_size: Model size string like "7B", "32B"

        Returns:
            Recommended quantization: "Q4_K_M", "Q5_K_M", "Q8_0", or "FP16"
        """
        size_match = re.match(r"(\d+)([BbMm])", model_size)
        if not size_match:
            return "Q4_K_M"

        num = int(size_match.group(1))
        unit = size_match.group(2).upper()
        params_b = num if unit == "B" else num / 1000

        usable_ram = self.ram_gb - 4

        # FP16: ~2GB per billion params
        if usable_ram >= params_b * 2 + 4:
            return "FP16"

        # Q8: ~1GB per billion params
        if usable_ram >= params_b * 1.0 + 3:
            return "Q8_0"

        # Q5: ~0.7GB per billion params
        if usable_ram >= params_b * 0.7 + 2:
            return "Q5_K_M"

        # Q4: ~0.5GB per billion params
        return "Q4_K_M"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hostname": self.hostname,
            "os": f"{self.os_name} {self.os_version}",
            "platform": self.platform,
            "cpu": {
                "name": self.cpu_name,
                "cores_physical": self.cpu_cores_physical,
                "cores_logical": self.cpu_cores_logical,
                "freq_mhz": self.cpu_freq_mhz,
            },
            "memory": {
                "total_gb": self.ram_gb,
                "available_gb": self.ram_available_gb,
                "unified": self.unified_memory,
                "bandwidth_gbps": self.memory_bandwidth_gbps,
            },
            "accelerator": {
                "type": self.chip_type.name,
                "name": self.chip_name,
                "gpu_cores": self.gpu_cores,
                "gpu_memory_gb": self.gpu_memory_gb,
                "neural_engine_cores": self.neural_engine_cores,
            },
            "capabilities": {
                "mps": self.supports_mps,
                "cuda": self.supports_cuda,
                "mlx": self.supports_mlx,
            },
        }


def detect_hardware() -> HardwareProfile:
    """
    Auto-detect hardware configuration.

    Returns:
        HardwareProfile with detected specifications.
    """
    profile = HardwareProfile()

    # Basic system info
    profile.hostname = platform.node()
    profile.os_name = platform.system()
    profile.os_version = platform.release()
    profile.platform = platform.machine()

    # CPU info
    profile.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
    profile.cpu_cores_logical = psutil.cpu_count(logical=True) or 0

    try:
        freq = psutil.cpu_freq()
        if freq:
            profile.cpu_freq_mhz = freq.current
    except Exception:
        pass

    # Memory
    mem = psutil.virtual_memory()
    profile.ram_gb = mem.total / (1024**3)
    profile.ram_available_gb = mem.available / (1024**3)

    # Platform-specific detection
    if platform.system() == "Darwin":
        _detect_apple_silicon(profile)
    elif platform.system() == "Linux":
        _detect_linux_gpu(profile)
    elif platform.system() == "Windows":
        _detect_windows_gpu(profile)

    # Check for MLX support (Apple Silicon only)
    if profile.chip_type == ChipType.APPLE_SILICON:
        try:
            import importlib.util

            profile.supports_mlx = importlib.util.find_spec("mlx.core") is not None
        except (ImportError, ModuleNotFoundError):
            profile.supports_mlx = False

    return profile


def _detect_apple_silicon(profile: HardwareProfile) -> None:
    """Detect Apple Silicon specifications."""
    try:
        # Get chip name via sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        profile.cpu_name = result.stdout.strip()

        # Check if Apple Silicon
        if "Apple" in profile.cpu_name:
            profile.chip_type = ChipType.APPLE_SILICON
            profile.chip_name = profile.cpu_name
            profile.unified_memory = True
            profile.supports_mps = True

            # Determine tier
            chip_lower = profile.cpu_name.lower()
            for tier in AppleSiliconTier:
                if tier.value != "unknown" and tier.value.replace("_", " ") in chip_lower:
                    profile.apple_tier = tier
                    break

            # Get GPU cores via system_profiler
            try:
                sp_result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                import json

                sp_data = json.loads(sp_result.stdout)
                displays = sp_data.get("SPDisplaysDataType", [])
                for display in displays:
                    if "sppci_cores" in display:
                        cores_str = display["sppci_cores"]
                        match = re.search(r"(\d+)", cores_str)
                        if match:
                            profile.gpu_cores = int(match.group(1))
                            break
            except Exception as e:
                logger.debug(f"Could not get GPU cores: {e}")

            # Get Neural Engine cores (approximation based on chip)
            ne_cores = {
                AppleSiliconTier.M1: 16,
                AppleSiliconTier.M1_PRO: 16,
                AppleSiliconTier.M1_MAX: 16,
                AppleSiliconTier.M1_ULTRA: 32,
                AppleSiliconTier.M2: 16,
                AppleSiliconTier.M2_PRO: 16,
                AppleSiliconTier.M2_MAX: 16,
                AppleSiliconTier.M2_ULTRA: 32,
                AppleSiliconTier.M3: 16,
                AppleSiliconTier.M3_PRO: 16,
                AppleSiliconTier.M3_MAX: 16,
                AppleSiliconTier.M4: 16,
                AppleSiliconTier.M4_PRO: 16,
                AppleSiliconTier.M4_MAX: 16,
            }
            profile.neural_engine_cores = ne_cores.get(profile.apple_tier, 16)

            # Memory bandwidth (approximation based on chip tier)
            bandwidth = {
                AppleSiliconTier.M1: 68.25,
                AppleSiliconTier.M1_PRO: 200,
                AppleSiliconTier.M1_MAX: 400,
                AppleSiliconTier.M1_ULTRA: 800,
                AppleSiliconTier.M2: 100,
                AppleSiliconTier.M2_PRO: 200,
                AppleSiliconTier.M2_MAX: 400,
                AppleSiliconTier.M2_ULTRA: 800,
                AppleSiliconTier.M3: 100,
                AppleSiliconTier.M3_PRO: 150,
                AppleSiliconTier.M3_MAX: 400,
                AppleSiliconTier.M4: 120,
                AppleSiliconTier.M4_PRO: 273,
                AppleSiliconTier.M4_MAX: 546,
            }
            profile.memory_bandwidth_gbps = bandwidth.get(profile.apple_tier, 100)

            # GPU memory = unified memory
            profile.gpu_memory_gb = profile.ram_gb

        else:
            # Intel Mac
            profile.chip_type = ChipType.INTEL

    except Exception as e:
        logger.warning(f"Apple Silicon detection failed: {e}")


def _detect_linux_gpu(profile: HardwareProfile) -> None:
    """Detect GPU on Linux systems."""
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                profile.chip_type = ChipType.NVIDIA
                profile.chip_name = parts[0]
                profile.gpu_memory_gb = float(parts[1].replace(" MiB", "")) / 1024
                profile.driver_version = parts[2]
                profile.supports_cuda = True

                # Get CUDA version
                cuda_result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if cuda_result.returncode == 0:
                    match = re.search(r"release (\d+\.\d+)", cuda_result.stdout)
                    if match:
                        profile.cuda_version = match.group(1)
                return
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"NVIDIA detection failed: {e}")

    # Try AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            profile.chip_type = ChipType.AMD
            # Parse ROCm output for GPU name
            for line in result.stdout.split("\n"):
                if "GPU" in line:
                    profile.chip_name = line.strip()
                    break
            return
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"AMD detection failed: {e}")

    # CPU only
    profile.chip_type = ChipType.CPU_ONLY
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    profile.cpu_name = line.split(":")[1].strip()
                    break
    except Exception:
        pass


def _detect_windows_gpu(profile: HardwareProfile) -> None:
    """Detect GPU on Windows systems."""
    # Try NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 2:
                profile.chip_type = ChipType.NVIDIA
                profile.chip_name = parts[0]
                profile.gpu_memory_gb = float(parts[1].replace(" MiB", "")) / 1024
                profile.supports_cuda = True
                return
    except Exception as e:
        logger.debug(f"NVIDIA detection failed: {e}")

    # Try WMI for GPU info
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [
                ln.strip()
                for ln in result.stdout.split("\n")
                if ln.strip() and ln.strip() != "Name"
            ]
            if lines:
                profile.chip_name = lines[0]
                if "NVIDIA" in profile.chip_name:
                    profile.chip_type = ChipType.NVIDIA
                elif "AMD" in profile.chip_name or "Radeon" in profile.chip_name:
                    profile.chip_type = ChipType.AMD
                elif "Intel" in profile.chip_name:
                    profile.chip_type = ChipType.INTEL
                return
    except Exception as e:
        logger.debug(f"WMI GPU detection failed: {e}")

    profile.chip_type = ChipType.CPU_ONLY


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent,
    }


def get_cpu_usage() -> Dict[str, float]:
    """Get current CPU usage."""
    return {
        "percent": psutil.cpu_percent(interval=0.1),
        "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
    }
