"""Project save/load helpers for configuration JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.core.config import ReconstructionParameters


def save_project_config(config: Dict[str, Any], file_path: str | Path) -> Path:
    """Save project configuration as JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return path


def load_project_config(file_path: str | Path) -> Dict[str, Any]:
    """Load project configuration from JSON and normalize tuple fields."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ReconstructionParameters.from_dict(data).to_dict()
