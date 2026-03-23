"""Tests for parameter validation logic."""

from __future__ import annotations

from src.core.validator import ParameterValidator


def build_valid_payload() -> dict[str, str]:
    return {
        "porosity": "0.35",
        "pore_size_distribution": "8, 12, 16",
        "specific_surface_area": "180",
        "coordination_number": "4",
        "reconstruction_size": "64,64,64",
        "seed": "42",
        "model_path": "assets/models/demo_generator.pt",
        "project_name": "测试项目",
    }


def test_validate_inputs_success_with_csv_distribution() -> None:
    result = ParameterValidator.validate_inputs(build_valid_payload())
    assert result.is_valid is True
    assert result.parsed_params is not None
    assert result.parsed_params["reconstruction_size"] == (64, 64, 64)


def test_validate_inputs_supports_json_distribution() -> None:
    payload = build_valid_payload()
    payload["pore_size_distribution"] = "[6, 10, 14]"
    result = ParameterValidator.validate_inputs(payload)
    assert result.is_valid is True
    assert result.parsed_params is not None
    assert result.parsed_params["pore_size_distribution"] == (6.0, 10.0, 14.0)


def test_validate_inputs_rejects_invalid_porosity() -> None:
    payload = build_valid_payload()
    payload["porosity"] = "1.5"
    result = ParameterValidator.validate_inputs(payload)
    assert result.is_valid is False
    assert "孔隙率" in result.error_message


def test_validate_inputs_rejects_invalid_size() -> None:
    payload = build_valid_payload()
    payload["reconstruction_size"] = "30,64,64"
    result = ParameterValidator.validate_inputs(payload)
    assert result.is_valid is False
    assert "8 的倍数" in result.error_message
