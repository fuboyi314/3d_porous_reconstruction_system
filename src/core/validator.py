"""Input validation helpers for reconstruction parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from src.core.config import ReconstructionParameters


@dataclass(slots=True)
class ValidationResult:
    """Structured result returned by parameter validation."""

    is_valid: bool
    parsed_params: dict[str, Any] | None
    error_message: str = ""


class ParameterValidator:
    """Validate and parse raw GUI input into structured parameters."""

    @staticmethod
    def parse_pore_size_distribution(raw_value: str) -> list[float]:
        """Parse pore-size distribution from CSV or simple JSON input."""
        text = raw_value.strip()
        if not text:
            raise ValueError("孔径分布不能为空。")

        if text.startswith("[") or text.startswith("{"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                values = payload.get("values")
                if values is None:
                    raise ValueError("JSON 孔径分布必须包含 values 字段。")
            else:
                values = payload
            if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                raise ValueError("孔径分布 JSON 必须是数值列表。")
            parsed = [float(item) for item in values]
        else:
            normalized = text.replace("；", ",").replace(";", ",")
            parsed = [float(item.strip()) for item in normalized.split(",") if item.strip()]

        if not parsed:
            raise ValueError("孔径分布解析后为空。")
        if any(value <= 0 for value in parsed):
            raise ValueError("孔径分布中的每个值都必须大于 0。")
        return parsed

    @staticmethod
    def parse_reconstruction_size(raw_value: str) -> tuple[int, int, int]:
        """Parse reconstruction size from comma- or x-separated text."""
        text = raw_value.strip().lower().replace("×", "x")
        for separator in ("x", ",", " "):
            if separator in text:
                parts = [part for part in text.replace("x", ",").replace(" ", ",").split(",") if part]
                break
        else:
            parts = [text]

        if len(parts) != 3:
            raise ValueError("重构尺寸必须输入为 3 个整数，例如 64,64,64。")

        try:
            size = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("重构尺寸必须全部为整数。") from exc

        return size  # type: ignore[return-value]

    @classmethod
    def validate_inputs(cls, raw_params: Dict[str, Any]) -> ValidationResult:
        """Validate raw GUI input and return structured parameters."""
        try:
            porosity = float(raw_params.get("porosity", ""))
            if not 0.0 <= porosity <= 1.0:
                raise ValueError("孔隙率必须位于 0 到 1 之间。")

            surface_area = float(raw_params.get("specific_surface_area", ""))
            if surface_area < 0:
                raise ValueError("比表面积必须为非负数。")

            coordination_number = float(raw_params.get("coordination_number", ""))
            if coordination_number < 0:
                raise ValueError("配位数必须为非负数。")

            reconstruction_size = cls.parse_reconstruction_size(str(raw_params.get("reconstruction_size", "")))
            if any(axis < 16 for axis in reconstruction_size):
                raise ValueError("重构尺寸的每个方向至少为 16。")
            if any(axis % 8 != 0 for axis in reconstruction_size):
                raise ValueError("重构尺寸必须为 8 的倍数。")

            seed = int(str(raw_params.get("seed", "")).strip())
            if seed < 0:
                raise ValueError("随机种子必须为非负整数。")

            pore_size_distribution = cls.parse_pore_size_distribution(str(raw_params.get("pore_size_distribution", "")))
            model_path = str(raw_params.get("model_path", "")).strip() or "assets/models/demo_generator.pt"
            project_name = str(raw_params.get("project_name", "基于神经网络的三维多孔介质重构系统")).strip()

            params = ReconstructionParameters(
                porosity=porosity,
                pore_size_distribution=tuple(pore_size_distribution),
                specific_surface_area=surface_area,
                coordination_number=coordination_number,
                reconstruction_size=reconstruction_size,
                seed=seed,
                model_path=model_path,
                project_name=project_name,
            )
            params.validate()
            return ValidationResult(is_valid=True, parsed_params=params.to_dict(), error_message="")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return ValidationResult(is_valid=False, parsed_params=None, error_message=str(exc))
