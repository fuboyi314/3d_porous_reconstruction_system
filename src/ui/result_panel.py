"""Result display widgets for slices, 3D placeholder, and textual summaries."""

from __future__ import annotations

from typing import Dict

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QTextEdit, QVBoxLayout, QWidget


class ResultPanel(QWidget):
    """Central result panel showing visualization placeholders and analysis text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slice_labels: Dict[str, QLabel] = {}
        self.result_text = QTextEdit(self)
        self.volume_placeholder = QLabel(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the middle result area requested by the specification."""
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._build_volume_group(), stretch=2)
        main_layout.addWidget(self._build_slice_group(), stretch=2)
        main_layout.addWidget(self._build_text_group(), stretch=1)

    def _build_volume_group(self) -> QGroupBox:
        """Create a placeholder for future 3D visualization widgets."""
        group = QGroupBox("三维显示占位区", self)
        layout = QVBoxLayout(group)
        self.volume_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.volume_placeholder.setWordWrap(True)
        self.volume_placeholder.setStyleSheet(
            "border: 1px dashed #7f8c8d; background-color: #f7f9fb; min-height: 220px; padding: 16px;"
        )
        self.volume_placeholder.setText(
            "当前版本采用稳定降级方案：此处保留三维显示接口。\n"
            "后续可替换为 PyVista/VTK 真实三维视图组件。"
        )
        layout.addWidget(self.volume_placeholder)
        return group

    def _build_slice_group(self) -> QGroupBox:
        """Create the X/Y/Z slice display area."""
        group = QGroupBox("三方向切片显示区", self)
        layout = QGridLayout(group)
        for index, axis_name in enumerate(("x", "y", "z")):
            label = QLabel(f"{axis_name.upper()} 方向切片", group)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "border: 1px solid #bdc3c7; background-color: white; min-height: 180px; padding: 8px;"
            )
            label.setScaledContents(False)
            self.slice_labels[axis_name] = label
            layout.addWidget(label, 0, index)
        return group

    def _build_text_group(self) -> QGroupBox:
        """Create the textual explanation output area."""
        group = QGroupBox("结果说明文本框", self)
        layout = QVBoxLayout(group)
        self.result_text.setReadOnly(True)
        self.result_text.setPlainText("重构完成后，结果说明将显示在这里。")
        layout.addWidget(self.result_text)
        return group

    def set_result_text(self, text: str) -> None:
        """Update the textual result description."""
        self.result_text.setPlainText(text)

    def set_volume_summary(self, text: str) -> None:
        """Update placeholder text with a compact volume summary."""
        self.volume_placeholder.setText(text)

    def update_slices(self, slices: Dict[str, np.ndarray]) -> None:
        """Render numpy slice arrays to Qt labels."""
        for axis_name, image_array in slices.items():
            if axis_name not in self.slice_labels:
                continue
            pixmap = self._numpy_to_pixmap(image_array)
            self.slice_labels[axis_name].setPixmap(pixmap)

    @staticmethod
    def _numpy_to_pixmap(image_array: np.ndarray) -> QPixmap:
        """Convert a 2D binary image array into a QPixmap."""
        normalized = (image_array.astype(np.uint8) * 255).copy(order="C")
        height, width = normalized.shape
        bytes_per_line = width
        qimage = QImage(normalized.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        qimage = qimage.copy()
        return QPixmap.fromImage(qimage).scaled(
            220,
            220,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
