"""Core configuration objects used throughout the application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_NAME = "基于神经网络的三维多孔介质重构系统"
APP_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
TESTS_DIR = REPO_ROOT / "tests"
PROJECTS_DIR = REPO_ROOT / "projects"

DEFAULT_DIRECTORIES = (
    ASSETS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    TESTS_DIR,
    PROJECTS_DIR,
)

PARAMETER_LIMITS: Dict[str, Tuple[float, float]] = {
    "porosity": (0.0, 1.0),
    "specific_surface_area": (0.0, 100000.0),
    "coordination_number": (0.0, 100.0),
    "size": (16.0, 256.0),
    "seed": (0.0, 2**31 - 1.0),
}


@dataclass(slots=True)
class ReconstructionParameters:
    """Structured porous-medium reconstruction parameters."""

    porosity: float = 0.35
    pore_size_distribution: tuple[float, ...] = (8.0, 12.0, 16.0)
    specific_surface_area: float = 180.0
    coordination_number: float = 4.0
    reconstruction_size: tuple[int, int, int] = (64, 64, 64)
    seed: int = 42
    model_path: str = "assets/models/demo_generator.pt"
    project_name: str = PROJECT_NAME

    def validate(self) -> None:
        """Validate parameter values used by the application."""
        if not (PARAMETER_LIMITS["porosity"][0] <= self.porosity <= PARAMETER_LIMITS["porosity"][1]):
            raise ValueError("孔隙率必须位于 0 到 1 之间。")

        if self.specific_surface_area < PARAMETER_LIMITS["specific_surface_area"][0]:
            raise ValueError("比表面积必须为非负数。")

        if self.coordination_number < PARAMETER_LIMITS["coordination_number"][0]:
            raise ValueError("配位数必须为非负数。")

        if len(self.reconstruction_size) != 3:
            raise ValueError("重构尺寸必须包含 3 个整数。")

        for axis_value in self.reconstruction_size:
            if not (PARAMETER_LIMITS["size"][0] <= axis_value <= PARAMETER_LIMITS["size"][1]):
                raise ValueError("重构尺寸必须位于 16 到 256 之间。")
            if axis_value % 8 != 0:
                raise ValueError("重构尺寸必须是 8 的倍数。")

        if self.seed < PARAMETER_LIMITS["seed"][0]:
            raise ValueError("随机种子必须为非负整数。")

        if not self.pore_size_distribution:
            raise ValueError("孔径分布不能为空。")

        if any(value <= 0 for value in self.pore_size_distribution):
            raise ValueError("孔径分布中的值必须为正数。")

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a JSON serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructionParameters":
        """Build parameters from a serialized dictionary."""
        if "pore_size_distribution" in data:
            data["pore_size_distribution"] = tuple(data["pore_size_distribution"])
        if "reconstruction_size" in data:
            data["reconstruction_size"] = tuple(data["reconstruction_size"])
        instance = cls(**data)
        instance.validate()
        return instance


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime configuration and directory layout."""

    app_name: str = PROJECT_NAME
    app_version: str = APP_VERSION
    repo_root: Path = field(default=REPO_ROOT)
    outputs_dir: Path = field(default=OUTPUTS_DIR)
    logs_dir: Path = field(default=LOGS_DIR)
    models_dir: Path = field(default=MODELS_DIR)
    projects_dir: Path = field(default=PROJECTS_DIR)

    def ensure_directories(self) -> None:
        """Create required runtime directories if missing."""
        for directory in DEFAULT_DIRECTORIES:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        """Expose runtime paths in plain string form."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "repo_root": str(self.repo_root),
            "outputs_dir": str(self.outputs_dir),
            "logs_dir": str(self.logs_dir),
            "models_dir": str(self.models_dir),
            "projects_dir": str(self.projects_dir),
        }


DEFAULT_PARAMETERS = ReconstructionParameters()
DEFAULT_RUNTIME_CONFIG = RuntimeConfig()
