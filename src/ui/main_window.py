"""Main application window integrating UI panels with business controllers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMainWindow, QMessageBox, QTextEdit, QVBoxLayout, QWidget

from src.core.config import PROJECT_NAME, RuntimeConfig
from src.core.task_controller import TaskController
from src.ui.parameter_panel import ParameterPanel
from src.ui.result_panel import ResultPanel


class QTextEditLogHandler(logging.Handler):
    """Forward log messages into a read-only QTextEdit widget."""

    def __init__(self, widget: QTextEdit) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.widget.append(message)


class MainWindow(QMainWindow):
    """Top-level main window with left parameter area, center results, bottom logs."""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        super().__init__()
        self.runtime_config = runtime_config
        self.runtime_config.ensure_directories()

        self.parameter_panel = ParameterPanel(self)
        self.result_panel = ResultPanel(self)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)

        self.logger = logging.getLogger("porous_reconstruction")
        self._configure_logging()
        self.controller = TaskController(runtime_config=self.runtime_config, logger=self.logger)

        self._build_ui()
        self._connect_signals()
        self.logger.info("软件启动成功。")

    def _build_ui(self) -> None:
        """Construct the main layout."""
        self.setWindowTitle(PROJECT_NAME)
        self.resize(1600, 960)

        central = QWidget(self)
        outer_layout = QVBoxLayout(central)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.parameter_panel, stretch=1)
        top_layout.addWidget(self.result_panel, stretch=2)
        outer_layout.addLayout(top_layout, stretch=4)
        outer_layout.addWidget(self._wrap_log_widget(), stretch=1)
        self.setCentralWidget(central)

    def _wrap_log_widget(self) -> QWidget:
        """Wrap the log editor so the bottom area is visually separated."""
        container = QWidget(self)
        layout = QVBoxLayout(container)
        self.log_text.setPlaceholderText("运行日志将在此处实时显示。")
        layout.addWidget(self.log_text)
        return container

    def _connect_signals(self) -> None:
        """Connect panel signals to controller-backed handlers."""
        self.parameter_panel.load_model_requested.connect(self.handle_load_model)
        self.parameter_panel.reconstruct_requested.connect(self.handle_reconstruct)
        self.parameter_panel.save_results_requested.connect(self.handle_save_project)
        self.parameter_panel.open_project_requested.connect(self.handle_open_project)
        self.parameter_panel.export_report_requested.connect(self.handle_export_report)
        self.parameter_panel.clear_requested.connect(self.handle_clear_parameters)

    def _configure_logging(self) -> None:
        """Configure file logging plus the GUI log view."""
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            Path(self.runtime_config.logs_dir) / "application.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        gui_handler = QTextEditLogHandler(self.log_text)
        gui_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(gui_handler)

    def handle_load_model(self, model_path: str) -> None:
        """Load the selected model path through the controller."""
        status = self.controller.load_model(model_path)
        self._show_status_message(status.success, status.message)

    def handle_reconstruct(self, raw_inputs: dict) -> None:
        """Run reconstruction and refresh the result panel."""
        status = self.controller.reconstruct(raw_inputs)
        if not status.success or status.payload is None:
            self._show_status_message(False, status.message)
            return

        analysis = status.payload["analysis"]
        params = status.payload["params"]
        slices = analysis["slices"]
        self.result_panel.update_slices(slices)
        self.result_panel.set_result_text(status.payload["report_text"])
        self.result_panel.set_volume_summary(
            f"三维重构已完成。\n"
            f"体素尺寸：{params['reconstruction_size']}\n"
            f"实际孔隙率：{analysis['actual_porosity']:.4f}\n"
            f"当前为稳定降级显示模式；后续可在此接入 PyVista/VTK 三维视图。"
        )
        self._show_status_message(True, status.message)

    def handle_save_project(self, raw_inputs: dict) -> None:
        """Save current project configuration as a JSON file."""
        default_name = raw_inputs.get("project_name", "porous_project") or "porous_project"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存项目",
            str(Path(self.runtime_config.projects_dir) / f"{default_name}.json"),
            "JSON 文件 (*.json)",
        )
        if not file_path:
            return
        status = self.controller.save_project(file_path, raw_inputs)
        self._show_status_message(status.success, status.message)

    def handle_open_project(self) -> None:
        """Open a project configuration and load it into the panel."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开项目",
            str(self.runtime_config.projects_dir),
            "JSON 文件 (*.json)",
        )
        if not file_path:
            return
        status = self.controller.open_project(file_path)
        if status.success and status.payload:
            self.parameter_panel.set_parameters(status.payload["params"])
        self._show_status_message(status.success, status.message)

    def handle_export_report(self) -> None:
        """Export reconstruction results to the outputs directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            str(self.runtime_config.outputs_dir),
        )
        if not directory:
            directory = str(self.runtime_config.outputs_dir)
        status = self.controller.export_report(directory)
        self._show_status_message(status.success, status.message)

    def handle_clear_parameters(self) -> None:
        """React to panel reset requests."""
        self.result_panel.set_result_text("参数已清空并恢复为默认值，等待重新重构。")
        self.logger.info("参数区已恢复默认值。")

    def _show_status_message(self, success: bool, message: str) -> None:
        """Display success or error messages consistently."""
        if success:
            QMessageBox.information(self, "提示", message)
        else:
            QMessageBox.warning(self, "提示", message)
