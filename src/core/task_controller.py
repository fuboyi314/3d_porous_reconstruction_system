"""Controller that coordinates GUI, model inference, analysis, and export tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.analysis.report_generator import analyze_volume, generate_result_description
from src.core.config import DEFAULT_PARAMETERS, RuntimeConfig
from src.core.validator import ParameterValidator, ValidationResult
from src.io.project_io import load_project_config, save_project_config
from src.io.result_exporter import ExportBundle, export_results
from src.model.inferencer import infer_structure
from src.model.model_loader import load_model


@dataclass(slots=True)
class TaskStatus:
    """Return value used by the GUI to inspect task execution status."""

    success: bool
    message: str
    payload: Dict[str, Any] | None = None


class TaskController:
    """Main application controller.

    This class keeps business logic out of the interface layer and provides a
    stable integration point for later replacing the demo generator with a real
    trained model.
    """

    def __init__(self, runtime_config: RuntimeConfig, logger: logging.Logger | None = None) -> None:
        self.runtime_config = runtime_config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.current_volume: np.ndarray | None = None
        self.current_params: Dict[str, Any] = DEFAULT_PARAMETERS.to_dict()
        self.current_analysis: Dict[str, Any] | None = None
        self.current_report_text: str = ""

    def validate_parameters(self, raw_inputs: Dict[str, Any]) -> ValidationResult:
        """Validate raw interface inputs."""
        result = ParameterValidator.validate_inputs(raw_inputs)
        if result.is_valid:
            self.logger.info("参数校验通过。")
        else:
            self.logger.warning("参数校验失败：%s", result.error_message)
        return result

    def load_model(self, model_path: str) -> TaskStatus:
        """Load or initialize a model from the given path."""
        try:
            self.model = load_model(model_path)
            self.current_params["model_path"] = model_path
            self.logger.info("模型加载完成：%s", model_path)
            return TaskStatus(True, f"模型加载成功：{model_path}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("模型加载失败：%s", exc)
            return TaskStatus(False, f"模型加载失败：{exc}")

    def reconstruct(self, raw_inputs: Dict[str, Any]) -> TaskStatus:
        """Validate inputs, run inference, analyze results, and cache them."""
        validation = self.validate_parameters(raw_inputs)
        if not validation.is_valid or validation.parsed_params is None:
            return TaskStatus(False, validation.error_message)

        parsed_params = validation.parsed_params
        try:
            if self.model is None or parsed_params["model_path"] != self.current_params.get("model_path"):
                self.model = load_model(parsed_params["model_path"])
                self.logger.info("重构前自动加载模型：%s", parsed_params["model_path"])

            self.logger.info("开始重构任务。")
            volume = infer_structure(parsed_params, self.model)
            analysis_results = analyze_volume(volume)
            report_text = generate_result_description(parsed_params, analysis_results)

            self.current_volume = volume
            self.current_params = parsed_params
            self.current_analysis = analysis_results
            self.current_report_text = report_text

            self.logger.info("重构任务完成。")
            return TaskStatus(
                True,
                "重构完成。",
                {
                    "volume": volume,
                    "analysis": analysis_results,
                    "report_text": report_text,
                    "params": parsed_params,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("重构任务失败：%s", exc)
            return TaskStatus(False, f"重构失败：{exc}")

    def save_project(self, file_path: str | Path, raw_inputs: Dict[str, Any]) -> TaskStatus:
        """Save current project configuration to JSON."""
        validation = self.validate_parameters(raw_inputs)
        if not validation.is_valid or validation.parsed_params is None:
            return TaskStatus(False, validation.error_message)

        try:
            path = save_project_config(validation.parsed_params, file_path)
            self.logger.info("项目配置已保存：%s", path)
            return TaskStatus(True, f"项目已保存：{path}", {"path": str(path)})
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("保存项目失败：%s", exc)
            return TaskStatus(False, f"保存项目失败：{exc}")

    def open_project(self, file_path: str | Path) -> TaskStatus:
        """Load project configuration from disk."""
        try:
            config = load_project_config(file_path)
            self.current_params = config
            self.logger.info("项目配置已打开：%s", file_path)
            return TaskStatus(True, "项目打开成功。", {"params": config})
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("打开项目失败：%s", exc)
            return TaskStatus(False, f"打开项目失败：{exc}")

    def export_report(self, base_output_dir: str | Path | None = None) -> TaskStatus:
        """Export current volume, report, slices, and logs."""
        if self.current_volume is None or self.current_analysis is None:
            return TaskStatus(False, "当前没有可导出的重构结果。")

        target_dir = Path(base_output_dir) if base_output_dir else self.runtime_config.outputs_dir
        project_name = str(self.current_params.get("project_name", "porous_project"))
        try:
            bundle: ExportBundle = export_results(
                volume=self.current_volume,
                config=self.current_params,
                analysis_results=self.current_analysis,
                report_text=self.current_report_text,
                log_source_dir=self.runtime_config.logs_dir,
                base_output_dir=target_dir,
                project_name=project_name,
            )
            self.logger.info("结果导出完成：%s", bundle.base_dir)
            return TaskStatus(True, f"结果导出完成：{bundle.base_dir}", {"bundle": bundle})
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("结果导出失败：%s", exc)
            return TaskStatus(False, f"结果导出失败：{exc}")
