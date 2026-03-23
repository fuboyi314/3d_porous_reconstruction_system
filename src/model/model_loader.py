"""Model loading utilities for the porous reconstruction application."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.model.generator import ConditionalVoxelGenerator, GeneratorConfig


def _create_demo_checkpoint(path: Path) -> ConditionalVoxelGenerator:
    """Create and persist a demo checkpoint for first-run experiences.

    The initial project version must be able to reconstruct a sample porous
    structure even when users have not prepared a real trained weight file yet.
    """
    model = ConditionalVoxelGenerator()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "meta": {"mode": "demo", "description": "Auto-created demo checkpoint."},
        },
        path,
    )
    model.eval()
    return model


def load_model(model_path: str) -> ConditionalVoxelGenerator:
    """Load a trained model if available, otherwise initialize a demo model.

    The saved file may either contain a plain state_dict or a dictionary with
    keys `state_dict` and optional `config`. If loading fails, the function
    falls back to generating a demo checkpoint so the GUI can still reconstruct
    a synthetic porous-medium sample on first run.
    """
    path = Path(model_path)

    if not path.exists():
        return _create_demo_checkpoint(path)

    try:
        checkpoint: Any = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            config_data = checkpoint.get("config") or {}
            model = ConditionalVoxelGenerator(GeneratorConfig(**config_data))
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model = ConditionalVoxelGenerator()
            model.load_state_dict(checkpoint)
        else:
            return _create_demo_checkpoint(path)
    except Exception:
        return _create_demo_checkpoint(path)

    model.eval()
    return model
