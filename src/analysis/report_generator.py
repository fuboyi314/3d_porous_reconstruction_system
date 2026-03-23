"""Generate structured analysis results and Chinese report text."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.analysis.coordination import estimate_coordination_number
from src.analysis.pore_size import analyze_connected_components, estimate_pore_size_statistics
from src.analysis.porosity import calculate_porosity, count_phase_voxels, extract_typical_slices
from src.analysis.surface_area import estimate_specific_surface_area


def analyze_volume(volume: np.ndarray) -> Dict[str, Any]:
    """Run the standard V1.0 analysis pipeline for a voxel structure."""
    phase_counts = count_phase_voxels(volume)
    connectivity = analyze_connected_components(volume)
    pore_size_stats = estimate_pore_size_statistics(volume)
    slices = extract_typical_slices(volume)

    return {
        "actual_porosity": calculate_porosity(volume),
        "phase_counts": phase_counts,
        "connectivity": connectivity,
        "specific_surface_area_estimate": estimate_specific_surface_area(volume),
        "coordination_number_estimate": estimate_coordination_number(volume),
        "pore_size_statistics": pore_size_stats,
        "slices": slices,
    }


def generate_result_description(input_params: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
    """Generate a concise Chinese explanation of reconstruction results."""
    size_x, size_y, size_z = input_params["reconstruction_size"]
    actual_porosity = analysis_results["actual_porosity"]
    target_porosity = input_params["porosity"]
    phase_counts = analysis_results["phase_counts"]
    connectivity = analysis_results["connectivity"]
    pore_stats = analysis_results["pore_size_statistics"]
    specific_surface_area = analysis_results["specific_surface_area_estimate"]
    coordination = analysis_results["coordination_number_estimate"]

    return (
        f"本次基于神经网络的三维多孔介质重构任务已完成，生成体素尺寸为 {size_x}×{size_y}×{size_z}。\n"
        f"输入目标孔隙率为 {target_porosity:.4f}，重构结果实际孔隙率为 {actual_porosity:.4f}。\n"
        f"孔隙体素数为 {phase_counts['pore_voxels']}，固体体素数为 {phase_counts['solid_voxels']}。\n"
        f"孔隙相连通域数量为 {connectivity['connected_components']}，最大连通域体素数为 {connectivity['largest_component_voxels']}。\n"
        f"估算比表面积为 {specific_surface_area:.4f}，估算平均配位数为 {coordination:.4f}。\n"
        f"等效平均孔径约为 {pore_stats['mean_equivalent_diameter']:.4f}，最大等效孔径约为 {pore_stats['max_equivalent_diameter']:.4f}。"
    )
