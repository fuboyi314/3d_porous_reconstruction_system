"""Result export utilities for reconstructed porous media projects."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.porosity import extract_typical_slices
from src.io.project_io import save_project_config


@dataclass(slots=True)
class ExportBundle:
    """Export path container for generated artifacts."""

    base_dir: Path
    config_path: Path
    volume_path: Path
    report_path: Path
    slice_dir: Path
    log_dir: Path


def prepare_export_bundle(base_output_dir: str | Path, project_name: str) -> ExportBundle:
    """Create a unified output directory structure."""
    base_dir = Path(base_output_dir) / project_name
    slice_dir = base_dir / "slices"
    log_dir = base_dir / "logs"
    slice_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return ExportBundle(
        base_dir=base_dir,
        config_path=base_dir / "config.json",
        volume_path=base_dir / "volume.npy",
        report_path=base_dir / "report.txt",
        slice_dir=slice_dir,
        log_dir=log_dir,
    )


def export_results(
    volume: np.ndarray,
    config: Dict[str, Any],
    analysis_results: Dict[str, Any],
    report_text: str,
    log_source_dir: str | Path,
    base_output_dir: str | Path,
    project_name: str,
) -> ExportBundle:
    """Export configuration, volume, report, slices, and log files."""
    bundle = prepare_export_bundle(base_output_dir, project_name)
    save_project_config(config, bundle.config_path)
    np.save(bundle.volume_path, volume)

    with bundle.report_path.open("w", encoding="utf-8") as file:
        file.write(report_text)

    slices = analysis_results.get("slices") or extract_typical_slices(volume)
    for axis_name, slice_array in slices.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(slice_array, cmap="gray", vmin=0, vmax=1)
        plt.title(f"{axis_name.upper()} 方向切片")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(bundle.slice_dir / f"slice_{axis_name}.png", dpi=150)
        plt.close()

    log_dir = Path(log_source_dir)
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            shutil.copy2(log_file, bundle.log_dir / log_file.name)

    return bundle
