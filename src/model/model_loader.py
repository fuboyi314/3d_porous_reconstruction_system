"""Model loading utilities for the porous reconstruction application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.model.generator import ConditionalVoxelGenerator, GeneratorConfig


def load_model(model_path: str) -> ConditionalVoxelGenerator:
    """Load a trained model if available, otherwise initialize a demo model.

    The saved file may either contain a plain state_dict or a dictionary with
    keys `state_dict` and optional `config`.
    """
    path = Path(model_path)
    model = ConditionalVoxelGenerator()

    if path.exists():
        checkpoint: Any = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            config_data = checkpoint.get("config")
            if config_data:
                model = ConditionalVoxelGenerator(GeneratorConfig(**config_data))
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "config": model.config.__dict__},
            path,
        )

    model.eval()
    return model
