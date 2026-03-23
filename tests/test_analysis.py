"""Tests for V1.0 analysis routines."""

from __future__ import annotations

import numpy as np

from src.analysis.pore_size import analyze_connected_components
from src.analysis.porosity import calculate_porosity, count_phase_voxels, extract_typical_slices
from src.analysis.report_generator import analyze_volume, generate_result_description
from src.analysis.surface_area import estimate_specific_surface_area


def build_test_volume() -> np.ndarray:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[2:6, 2:6, 2:6] = 1
    return volume


def test_calculate_porosity_and_counts() -> None:
    volume = build_test_volume()
    porosity = calculate_porosity(volume)
    counts = count_phase_voxels(volume)
    assert porosity == 64 / 512
    assert counts["pore_voxels"] == 64
    assert counts["solid_voxels"] == 448


def test_extract_typical_slices() -> None:
    slices = extract_typical_slices(build_test_volume())
    assert set(slices.keys()) == {"x", "y", "z"}
    assert slices["x"].shape == (8, 8)


def test_connected_components_and_surface_area() -> None:
    volume = build_test_volume()
    connectivity = analyze_connected_components(volume)
    surface_area = estimate_specific_surface_area(volume)
    assert connectivity["connected_components"] == 1
    assert connectivity["largest_component_voxels"] == 64
    assert surface_area > 0


def test_report_generation_contains_chinese_summary() -> None:
    volume = build_test_volume()
    analysis = analyze_volume(volume)
    params = {
        "porosity": 0.2,
        "reconstruction_size": (8, 8, 8),
    }
    report = generate_result_description(params, analysis)
    assert "实际孔隙率" in report
    assert "孔隙体素数" in report
