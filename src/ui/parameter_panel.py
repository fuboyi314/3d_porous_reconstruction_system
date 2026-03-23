"""Parameter input panel separated from the main window for maintainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import DEFAULT_PARAMETERS


class ParameterPanel(QWidget):
    """Left-side input panel containing parameter fields and action buttons."""

    load_model_requested = Signal(str)
    reconstruct_requested = Signal(dict)
    save_results_requested = Signal(dict)
    open_project_requested = Signal()
    export_report_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.inputs: Dict[str, QLineEdit] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the form controls and button area."""
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._build_parameter_group())
        main_layout.addWidget(self._build_button_group())
        main_layout.addStretch(1)

    def _build_parameter_group(self) -> QGroupBox:
        """Create all required parameter input widgets."""
        defaults = DEFAULT_PARAMETERS
        group = QGroupBox("参数输入区", self)
        layout = QFormLayout(group)

        fields = {
            "porosity": str(defaults.porosity),
            "pore_size_distribution": ", ".join(str(item) for item in defaults.pore_size_distribution),
            "specific_surface_area": str(defaults.specific_surface_area),
            "coordination_number": str(defaults.coordination_number),
            "reconstruction_size": ",".join(str(item) for item in defaults.reconstruction_size),
            "seed": str(defaults.seed),
            "project_name": defaults.project_name,
        }

        labels = {
            "porosity": "孔隙率",
            "pore_size_distribution": "孔径分布",
            "specific_surface_area": "比表面积",
            "coordination_number": "配位数",
            "reconstruction_size": "重构尺寸",
            "seed": "随机种子",
            "project_name": "项目名称",
        }

        for key, value in fields.items():
            widget = QLineEdit(value, group)
            self.inputs[key] = widget
            layout.addRow(labels[key], widget)

        model_path_layout = QHBoxLayout()
        model_path_edit = QLineEdit(defaults.model_path, group)
        browse_button = QPushButton("浏览", group)
        browse_button.clicked.connect(self._browse_model_path)
        model_path_layout.addWidget(model_path_edit)
        model_path_layout.addWidget(browse_button)
        self.inputs["model_path"] = model_path_edit
        layout.addRow("模型路径", model_path_layout)

        return group

    def _build_button_group(self) -> QGroupBox:
        """Create the panel button block without embedding business logic."""
        group = QGroupBox("控制按钮区", self)
        layout = QGridLayout(group)

        buttons = [
            ("加载模型", self._emit_load_model),
            ("开始重构", self._emit_reconstruct),
            ("保存结果", self._emit_save_results),
            ("打开项目", self.open_project_requested.emit),
            ("导出报告", self.export_report_requested.emit),
            ("清空参数", self._clear_parameters),
        ]

        for index, (label, callback) in enumerate(buttons):
            button = QPushButton(label, group)
            button.clicked.connect(callback)
            layout.addWidget(button, index // 2, index % 2)

        return group

    def _browse_model_path(self) -> None:
        """Open a file dialog to select the model path."""
        current = self.inputs["model_path"].text().strip() or str(Path.cwd())
        selected_file, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            current,
            "PyTorch 模型 (*.pt *.pth);;所有文件 (*)",
        )
        if selected_file:
            self.inputs["model_path"].setText(selected_file)

    def get_parameters(self) -> Dict[str, Any]:
        """Return raw parameter text values for validation by the controller."""
        return {name: widget.text().strip() for name, widget in self.inputs.items()}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Populate the panel from a previously saved configuration."""
        for key, value in params.items():
            if key not in self.inputs:
                continue
            if key == "pore_size_distribution" and isinstance(value, (list, tuple)):
                text = ", ".join(str(item) for item in value)
            elif key == "reconstruction_size" and isinstance(value, (list, tuple)):
                text = ",".join(str(item) for item in value)
            else:
                text = str(value)
            self.inputs[key].setText(text)

    def _emit_load_model(self) -> None:
        """Emit a model-loading request with the current path."""
        self.load_model_requested.emit(self.inputs["model_path"].text().strip())

    def _emit_reconstruct(self) -> None:
        """Emit a reconstruction request with all raw parameters."""
        self.reconstruct_requested.emit(self.get_parameters())

    def _emit_save_results(self) -> None:
        """Emit a save request with the current parameter set."""
        self.save_results_requested.emit(self.get_parameters())

    def _clear_parameters(self) -> None:
        """Reset the panel to default values and notify the main window."""
        self.set_parameters(DEFAULT_PARAMETERS.to_dict())
        self.clear_requested.emit()
