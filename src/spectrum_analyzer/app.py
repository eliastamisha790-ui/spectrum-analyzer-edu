"""PyQt5 desktop interface for Spectrum Analyzer."""

from __future__ import annotations

import datetime
import json
import os
import sys
import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .core import (
    ActivitySummary,
    AnalysisConfig,
    BackgroundInfo,
    CalibrationModel,
    EfficiencyModel,
    EfficiencyPoint,
    NNDC_LIBRARY,
    NuclideMatchSummary,
    PeakResult,
    SpectrumRecord,
    SPECTRUM_FILE_FILTER,
    analyze_spectrum,
    auto_calibrate,
    detect_peaks_from_cwt,
    decompose_interference,
    ensure_directory,
    fit_energy_calibration,
    flatten_library,
    format_activity,
    load_nuclide_library,
    load_spectrum_file,
    match_nuclides,
    merge_close_peaks,
    refine_peaks,
    smooth_spectrum,
    snip_baseline,
    subtract_background,
    summarize_nuclide_matches,
    validate_nuclide_library,
)
from .demo import DEMO_OFFSET_KEV, DEMO_SLOPE_KEV_PER_CHANNEL, synthetic_co60_record
from .certificate import certificate_library_from_text

APP_DATA_DIR = Path(os.environ.get("SPECTRUM_ANALYZER_DATA_DIR", Path.home() / ".spectrum-analyzer"))
REPORT_DIR = str(APP_DATA_DIR / "reports")
LIBRARY_DIR = str(APP_DATA_DIR / "libraries")
EFFICIENCY_CACHE_PATH = str(APP_DATA_DIR / "efficiency_model.json")
BASE_DIR = str(Path(__file__).resolve().parent)

CERTIFICATE_TEMPLATE = """标准源：GBW08304a（300g）
核素比活度 (Bq/kg)
238U：158
232Th：65.6
226Ra：132
40K：535
60Co：137
137Cs：81
定值时间：1995年1月31日
"""


class CertificateLibraryDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, base_library: Dict[str, List[Dict[str, float]]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("从证书生成核素库")
        self.setModal(True)
        self.resize(760, 560)
        self.base_library = base_library
        self.result = None
        layout = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "把证书上的核素与比活度粘贴到下面（一行一个，兼容“238U：158 Bq/kg”“Cs-137 81”等写法）。\n"
            "软件只保留证书能解释的谱线，并按子体展开：238U→234Th/235U，232Th→228Ac/212Pb/208Tl，226Ra→186 keV/214Pb/214Bi。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)
        self.editor = QtWidgets.QPlainTextEdit(CERTIFICATE_TEMPLATE)
        self.editor.setFont(QtGui.QFont("Consolas", 10))
        layout.addWidget(self.editor, 1)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        self.preview.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.preview, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("生成并应用")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.editor.textChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        try:
            self.result = certificate_library_from_text(self.editor.toPlainText(), self.base_library)
        except ValueError as exc:
            self.result = None
            self.preview.setText(f"⚠ {exc}")
            return
        self.preview.setText(self.result.summary_text())

    def accept(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "无法生成", "证书内容还不能解析出核素，请检查格式。")
            return
        super().accept()


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, isotopes: List[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择标准源")
        self.setModal(True)
        self.resize(360, 180)
        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel("请选择用于能量刻度的标准源核素。\n软件将自动寻峰并拟合多项式。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(isotopes)
        layout.addWidget(self.combo)
        self.accept_btn = QtWidgets.QPushButton("开始刻度")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.accept_btn)
        layout.addLayout(btn_layout)
        self.accept_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def selected_isotope(self) -> str:
        return self.combo.currentText()


class ManualCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("自定义能量刻度点")
        self.setModal(True)
        self.resize(460, 360)
        layout = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "每行输入一个“峰道址, 已知能量(keV)”刻度点。\n"
            "示例：9062.8, 1173.237。刻度点应来自可靠标准源并由用户确认。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)
        self.points_edit = QtWidgets.QPlainTextEdit()
        self.points_edit.setPlaceholderText("9062.8, 1173.237\n10293.6, 1332.501")
        layout.addWidget(self.points_edit, 1)
        form = QtWidgets.QFormLayout()
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 2)
        self.degree_spin.setValue(1)
        form.addRow("拟合阶数", self.degree_spin)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._channels: List[float] = []
        self._energies: List[float] = []

    def accept(self) -> None:
        channels: List[float] = []
        energies: List[float] = []
        try:
            for line_number, line in enumerate(self.points_edit.toPlainText().splitlines(), 1):
                if not line.strip():
                    continue
                parts = [item.strip() for item in line.replace("，", ",").split(",")]
                if len(parts) != 2:
                    raise ValueError(f"第 {line_number} 行应包含两个逗号分隔的数值。")
                channels.append(float(parts[0]))
                energies.append(float(parts[1]))
            degree = int(self.degree_spin.value())
            fit_energy_calibration(channels, energies, degree=degree)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "刻度点无效", str(exc))
            return
        self._channels = channels
        self._energies = energies
        super().accept()

    def values(self) -> Tuple[List[float], List[float], int]:
        return self._channels, self._energies, int(self.degree_spin.value())


class AnalysisSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, config: AnalysisConfig) -> None:
        super().__init__(parent)
        self.setWindowTitle("分析参数")
        self.setModal(True)
        self.resize(520, 500)
        layout = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel("参数需要根据探测器分辨率、道数和计数统计调整；默认值适合一般教学示例。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        form = QtWidgets.QFormLayout()

        self.snip_spin = QtWidgets.QSpinBox()
        self.snip_spin.setRange(0, 4096)
        self.snip_spin.setSpecialValueText("自动")
        self.snip_spin.setValue(config.snip_iterations or 0)
        form.addRow("SNIP 迭代", self.snip_spin)

        self.widths_edit = QtWidgets.QLineEdit(",".join(str(value) for value in config.wavelet_widths))
        form.addRow("小波宽度(通道)", self.widths_edit)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 20.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(config.detection_threshold_sigma)
        form.addRow("寻峰阈值系数", self.threshold_spin)

        self.distance_spin = QtWidgets.QSpinBox()
        self.distance_spin.setRange(1, 4096)
        self.distance_spin.setValue(config.min_peak_distance)
        form.addRow("最小峰间距(通道)", self.distance_spin)

        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(0.0, 1000.0)
        self.snr_spin.setDecimals(2)
        self.snr_spin.setValue(config.min_snr)
        form.addRow("最小信噪比", self.snr_spin)

        self.absolute_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.absolute_tolerance_spin.setRange(0.01, 1000.0)
        self.absolute_tolerance_spin.setDecimals(3)
        self.absolute_tolerance_spin.setValue(config.absolute_tolerance_kev)
        form.addRow("绝对匹配容差(keV)", self.absolute_tolerance_spin)

        self.relative_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.relative_tolerance_spin.setRange(0.0, 100.0)
        self.relative_tolerance_spin.setDecimals(4)
        self.relative_tolerance_spin.setValue(config.relative_tolerance * 100.0)
        form.addRow("相对匹配容差(%)", self.relative_tolerance_spin)

        self.candidates_spin = QtWidgets.QSpinBox()
        self.candidates_spin.setRange(1, 20)
        self.candidates_spin.setValue(config.max_candidates)
        form.addRow("每峰候选上限", self.candidates_spin)

        self.width_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.width_tolerance_spin.setRange(0.0, 5.0)
        self.width_tolerance_spin.setSingleStep(0.1)
        self.width_tolerance_spin.setDecimals(2)
        self.width_tolerance_spin.setValue(config.width_tolerance_factor)
        form.addRow("容差按 FWHM 放宽(倍)", self.width_tolerance_spin)

        self.interference_spin = QtWidgets.QDoubleSpinBox()
        self.interference_spin.setRange(0.0, 10.0)
        self.interference_spin.setSingleStep(0.1)
        self.interference_spin.setDecimals(2)
        self.interference_spin.setValue(config.interference_width_factor)
        form.addRow("双线判定宽度(倍 FWHM)", self.interference_spin)

        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._config = config

    def _restore_defaults(self) -> None:
        defaults = AnalysisConfig()
        self.snip_spin.setValue(0)
        self.widths_edit.setText(",".join(str(value) for value in defaults.wavelet_widths))
        self.threshold_spin.setValue(defaults.detection_threshold_sigma)
        self.distance_spin.setValue(defaults.min_peak_distance)
        self.snr_spin.setValue(defaults.min_snr)
        self.absolute_tolerance_spin.setValue(defaults.absolute_tolerance_kev)
        self.relative_tolerance_spin.setValue(defaults.relative_tolerance * 100.0)
        self.candidates_spin.setValue(defaults.max_candidates)
        self.width_tolerance_spin.setValue(defaults.width_tolerance_factor)
        self.interference_spin.setValue(defaults.interference_width_factor)

    def accept(self) -> None:
        try:
            widths = tuple(int(item.strip()) for item in self.widths_edit.text().split(",") if item.strip())
            self._config = AnalysisConfig(
                snip_iterations=self.snip_spin.value() or None,
                wavelet_widths=widths,
                detection_threshold_sigma=self.threshold_spin.value(),
                min_peak_distance=self.distance_spin.value(),
                min_snr=self.snr_spin.value(),
                absolute_tolerance_kev=self.absolute_tolerance_spin.value(),
                relative_tolerance=self.relative_tolerance_spin.value() / 100.0,
                max_candidates=self.candidates_spin.value(),
                width_tolerance_factor=self.width_tolerance_spin.value(),
                interference_width_factor=self.interference_spin.value(),
                background_scale=self._config.background_scale,
                snip_after_background=self._config.snip_after_background,
            ).validated()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(exc))
            return
        super().accept()

    def config(self) -> AnalysisConfig:
        return self._config


class EfficiencyCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, isotopes: List[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("效率刻度设置")
        self.setModal(True)
        self.resize(420, 240)
        layout = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "选择已知活度的标准源，输入活度及刻度拟合阶数。\n"
            "系统将利用所选峰的净面积与γ支路强度计算效率。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.iso_combo = QtWidgets.QComboBox()
        self.iso_combo.addItems(isotopes)
        form.addRow("标准源", self.iso_combo)

        activity_layout = QtWidgets.QHBoxLayout()
        self.activity_input = QtWidgets.QDoubleSpinBox()
        self.activity_input.setDecimals(6)
        self.activity_input.setRange(1e-6, 1e12)
        self.activity_input.setValue(1.0)
        activity_layout.addWidget(self.activity_input, 2)
        self.activity_unit = QtWidgets.QComboBox()
        self.activity_unit.addItem("Bq", 1.0)
        self.activity_unit.addItem("kBq", 1e3)
        self.activity_unit.addItem("MBq", 1e6)
        self.activity_unit.addItem("GBq", 1e9)
        activity_layout.addWidget(self.activity_unit, 1)
        form.addRow("活度", activity_layout)

        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 4)
        self.degree_spin.setValue(2)
        form.addRow("拟合阶数", self.degree_spin)

        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.ok_btn = QtWidgets.QPushButton("开始刻度")
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
        layout.addLayout(btn_layout)

        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.accept)

    def selected_isotope(self) -> str:
        return self.iso_combo.currentText()

    def activity_bq(self) -> float:
        factor = float(self.activity_unit.currentData() or 1.0)
        return self.activity_input.value() * factor

    def degree(self) -> int:
        return int(self.degree_spin.value())


class PeakSelectionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        peaks: List[PeakResult],
        calibration: CalibrationModel,
        title: str,
        tip: str,
        min_required: int = 1,
        preview_callback: Optional[Callable[[PeakResult], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setModal(False)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.resize(540, 460)
        self._peaks = peaks
        self._calibration = calibration
        self._min_required = max(1, min_required)
        self._preview_callback = preview_callback

        layout = QtWidgets.QVBoxLayout(self)
        tip_label = QtWidgets.QLabel(tip)
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)

        self.table = QtWidgets.QTableWidget(len(peaks), 6)
        self.table.setHorizontalHeaderLabels(["选择", "道址", "能量(keV)", "峰面积", "半高宽", "信噪比"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.table)

        for row, peak in enumerate(peaks):
            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            check_item.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(row, 0, check_item)

            channel_item = QtWidgets.QTableWidgetItem(f"{peak.channel:.1f}")
            channel_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 1, channel_item)

            energy = calibration.energy(peak.channel)
            energy_item = QtWidgets.QTableWidgetItem(f"{energy:.2f}")
            energy_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 2, energy_item)

            area_item = QtWidgets.QTableWidgetItem(f"{peak.area:.1f}")
            area_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 3, area_item)

            fwhm_item = QtWidgets.QTableWidgetItem(f"{peak.fwhm:.2f}")
            fwhm_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 4, fwhm_item)

            snr_item = QtWidgets.QTableWidgetItem(f"{peak.snr:.2f}")
            snr_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 5, snr_item)

        self.counter_label = QtWidgets.QLabel()
        layout.addWidget(self.counter_label)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.ok_btn = QtWidgets.QPushButton("确认")
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
        layout.addLayout(btn_layout)

        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.accept)
        self.table.itemChanged.connect(self._update_counter)
        self.table.itemSelectionChanged.connect(self._emit_preview)
        self._update_counter()
        if peaks:
            self.table.selectRow(0)
            QtCore.QTimer.singleShot(0, self._emit_preview)

    def _emit_preview(self) -> None:
        if not self._preview_callback:
            return
        row = self.table.currentRow()
        if 0 <= row < len(self._peaks):
            self._preview_callback(self._peaks[row])

    def _update_counter(self) -> None:
        selected = sum(
            1 for row in range(self.table.rowCount()) if self.table.item(row, 0).checkState() == QtCore.Qt.Checked
        )
        self.counter_label.setText(f"已选择峰 {selected} 个 / 至少 {self._min_required} 个")

    def selected_peaks(self) -> List[PeakResult]:
        chosen: List[PeakResult] = []
        for row, peak in enumerate(self._peaks):
            if self.table.item(row, 0).checkState() == QtCore.Qt.Checked:
                chosen.append(peak)
        return chosen

    def accept(self) -> None:
        if len(self.selected_peaks()) < self._min_required:
            QtWidgets.QMessageBox.warning(self, "提示", f"至少选 {self._min_required} 个峰。")
            return
        super().accept()


class AxisLockedViewBox(pg.ViewBox):
    """支持切换缩放模式的 ViewBox。"""

    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self._mode = "locked"  # locked / x / y / free
        self.setMouseMode(self.RectMode)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "x":
            self.setMouseEnabled(x=True, y=False)
            self.setAspectLocked(False)
        elif mode == "y":
            self.setMouseEnabled(x=False, y=True)
            self.setAspectLocked(False)
        elif mode == "free":
            self.setMouseEnabled(x=True, y=True)
            self.setAspectLocked(False)
        else:  # locked
            self.setMouseEnabled(x=True, y=True)
            self.setAspectLocked(True)

    def wheelEvent(self, ev, axis=None):  # type: ignore[override]
        prev = self.state["mouseEnabled"][:]
        self._apply_mouse_state()
        super().wheelEvent(ev, axis=axis)
        self.state["mouseEnabled"] = prev

    def mouseDragEvent(self, ev, axis=None):  # type: ignore[override]
        prev = self.state["mouseEnabled"][:]
        self._apply_mouse_state()
        super().mouseDragEvent(ev, axis=axis)
        self.state["mouseEnabled"] = prev

    def _apply_mouse_state(self) -> None:
        if self._mode == "x":
            self.state["mouseEnabled"] = [True, False]
        elif self._mode == "y":
            self.state["mouseEnabled"] = [False, True]
        else:
            self.state["mouseEnabled"] = [True, True]


class SpectrumAnalyzerApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("能谱分析工作台 v0.7")
        self.resize(1280, 760)
        self.current_spectrum: Optional[SpectrumRecord] = None
        self.background_spectrum: Optional[SpectrumRecord] = None
        self.background_info: Optional[BackgroundInfo] = None
        self.net_counts: Optional[np.ndarray] = None
        self.analysis_counts: Optional[np.ndarray] = None
        self.smoothed_counts: Optional[np.ndarray] = None
        self.smoothed_net_counts: Optional[np.ndarray] = None
        self.peaks: List[PeakResult] = []
        self.calibration = CalibrationModel()
        self.calibration_source = "默认（未刻度）"
        self.efficiency_model = EfficiencyModel()
        self.efficiency_points: List[EfficiencyPoint] = []
        self.activity_summary: List[ActivitySummary] = []
        self.nuclide_summary: List[NuclideMatchSummary] = []
        self.activity_status_message: str = "等待效率刻度"
        self.nuclide_library = validate_nuclide_library(NNDC_LIBRARY)
        self.library_source = "内置教学库"
        self.analysis_config = AnalysisConfig()
        self.flat_library = flatten_library(self.nuclide_library)
        self._peak_dialog: Optional[PeakSelectionDialog] = None
        self.peak_labels: List[pg.TextItem] = []
        self.view_box: Optional[AxisLockedViewBox] = None
        self._data_bounds: Optional[Tuple[float, float, float, float]] = None
        self._setup_ui()
        self._load_saved_efficiency_model()

    # ---------------- UI 构建 ----------------
    def _setup_ui(self) -> None:
        self._setup_menus()
        pg.setConfigOption("background", "#0b0d15")
        pg.setConfigOption("foreground", "#d8dee9")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setObjectName("ribbonBar")
        self.toolbar.setStyleSheet(
            """
            QFrame#ribbonBar {
                background-color: #161a27;
                border-radius: 10px;
            }
            QFrame#ribbonBar QLabel {
                color: #f5f5ff;
            }
            QFrame#ribbonBar QPushButton {
                background-color: #2c3250;
                border: 1px solid #3a3f63;
                border-radius: 6px;
                padding: 6px 14px;
                color: #f8f8ff;
            }
            QFrame#ribbonBar QPushButton:hover {
                background-color: #3d4366;
            }
            """
        )
        ribbon_layout = QtWidgets.QHBoxLayout(self.toolbar)
        ribbon_layout.setContentsMargins(16, 10, 16, 10)
        ribbon_layout.setSpacing(16)

        info_block = QtWidgets.QVBoxLayout()
        info_title = QtWidgets.QLabel("当前谱图")
        info_title.setStyleSheet("font-weight: 600; color: #a5b4fc;")
        self.info_label = QtWidgets.QLabel("当前未加载谱文件")
        self.info_label.setWordWrap(True)
        info_block.addWidget(info_title)
        info_block.addWidget(self.info_label)
        ribbon_layout.addLayout(info_block, 2)

        self.load_btn = QtWidgets.QPushButton("导入谱文件")
        self.background_btn = QtWidgets.QPushButton("导入本底谱")
        self.demo_btn = QtWidgets.QPushButton("加载合成示例")
        self.calibrate_btn = QtWidgets.QPushButton("能量刻度")
        self.efficiency_btn = QtWidgets.QPushButton("效率刻度")
        self.analyze_btn = QtWidgets.QPushButton("自动分析")
        self.manual_peaks_btn = QtWidgets.QPushButton("人工确认峰")
        self.report_btn = QtWidgets.QPushButton("导出报告")
        buttons = (
            self.load_btn,
            self.background_btn,
            self.demo_btn,
            self.calibrate_btn,
            self.efficiency_btn,
            self.analyze_btn,
            self.manual_peaks_btn,
            self.report_btn,
        )
        for btn in buttons:
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.setSpacing(10)
        for btn in buttons:
            buttons_row.addWidget(btn)
        buttons_row.addStretch(1)
        ribbon_layout.addLayout(buttons_row, 3)

        zoom_block = QtWidgets.QVBoxLayout()
        zoom_block.setSpacing(4)
        zoom_title = QtWidgets.QLabel("缩放模式")
        zoom_title.setStyleSheet("font-weight: 600; color: #a5b4fc;")
        self.zoom_combo = QtWidgets.QComboBox()
        self.zoom_combo.addItem("自由缩放", "free")
        self.zoom_combo.addItem("仅 X 轴", "x")
        self.zoom_combo.addItem("仅 Y 轴", "y")
        self.zoom_combo.addItem("锁定比例", "locked")
        self.zoom_combo.setFixedWidth(120)
        self.reset_view_btn = QtWidgets.QPushButton("重置视图")
        zoom_block.addWidget(zoom_title)
        zoom_block.addWidget(self.zoom_combo)
        zoom_block.addWidget(self.reset_view_btn)
        ribbon_layout.addLayout(zoom_block, 1)
        main_layout.addWidget(self.toolbar)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(12)

        plot_container = QtWidgets.QFrame()
        plot_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)

        self.view_box = AxisLockedViewBox()
        self.view_box.set_mode("free")
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, enableMenu=False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", "道址")
        self.plot_widget.setLabel("left", "计数")
        self.plot_widget.addLegend()
        self.view_box = self.plot_widget.getViewBox()
        self.raw_curve = self.plot_widget.plot([], [], pen=pg.mkPen("#5ac8fa", width=1.2), name="原始谱")
        self.smooth_curve = self.plot_widget.plot([], [], pen=pg.mkPen("#30d158", width=1.1), name="平滑辅助")
        self.net_curve = self.plot_widget.plot([], [], pen=pg.mkPen("#ff9500", width=1.0, style=QtCore.Qt.DashLine), name="净谱")
        self.background_curve = self.plot_widget.plot(
            [], [], pen=pg.mkPen("#9a9aa8", width=1.0, style=QtCore.Qt.DotLine), name="本底谱"
        )
        self.smooth_curve.hide()
        self.net_curve.hide()
        self.background_curve.hide()
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff80"))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#ffffff40"))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        plot_layout.addWidget(self.plot_widget, 1)
        self.cursor_label = QtWidgets.QLabel("光标: —")
        self.cursor_label.setStyleSheet("color: #cfd8ff;")
        plot_layout.addWidget(self.cursor_label)
        content_layout.addWidget(plot_container, 2)

        side_panel = QtWidgets.QFrame()
        side_panel.setFixedWidth(620)
        side_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)

        smoothing_group = QtWidgets.QGroupBox("谱光滑辅助")
        smoothing_layout = QtWidgets.QVBoxLayout(smoothing_group)
        self.smoothing_enable_check = QtWidgets.QCheckBox("分析时启用平滑辅助")
        self.smoothing_enable_check.setChecked(False)
        smoothing_layout.addWidget(self.smoothing_enable_check)
        smoothing_row = QtWidgets.QHBoxLayout()
        smoothing_row.addWidget(QtWidgets.QLabel("窗口"))
        self.smoothing_window_spin = QtWidgets.QSpinBox()
        self.smoothing_window_spin.setRange(3, 51)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(7)
        self.smoothing_window_spin.setEnabled(False)
        smoothing_row.addWidget(self.smoothing_window_spin, 1)
        smoothing_layout.addLayout(smoothing_row)
        side_layout.addWidget(smoothing_group)

        self.peaks_table = QtWidgets.QTableWidget(0, 7)
        self.peaks_table.setHorizontalHeaderLabels(["峰道址", "能量(keV)", "峰面积", "半高宽", "核素候选", "匹配分数", "活度估算"])
        self.peaks_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for column, width in enumerate((60, 78, 70, 64, 150, 74, 64)):
            self.peaks_table.setColumnWidth(column, width)
        self.peaks_table.setMinimumHeight(155)
        self.peaks_table.verticalHeader().setVisible(False)
        self.peaks_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.peaks_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table_font = QtGui.QFont(self.peaks_table.font())
        table_font.setPointSize(max(table_font.pointSize() - 1, 9))
        self.peaks_table.setFont(table_font)
        self.peaks_table.horizontalHeader().setFont(table_font)
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.setStyleSheet("QTableWidget { gridline-color: #2f354d; }")
        self.peaks_table.itemSelectionChanged.connect(self._handle_table_selection)
        side_layout.addWidget(self.peaks_table, 1)

        identification_group = QtWidgets.QGroupBox("整谱核素候选（启发式）")
        identification_group.setMaximumHeight(185)
        identification_layout = QtWidgets.QVBoxLayout(identification_group)
        self.identification_table = QtWidgets.QTableWidget(0, 4)
        self.identification_table.setHorizontalHeaderLabels(["核素", "匹配线", "强度覆盖", "谱级分数"])
        self.identification_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.identification_table.verticalHeader().setVisible(False)
        self.identification_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.identification_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.identification_table.setAlternatingRowColors(True)
        identification_layout.addWidget(self.identification_table)
        side_layout.addWidget(identification_group)

        activity_group = QtWidgets.QGroupBox("活度分析")
        activity_group.setMaximumHeight(190)
        activity_layout = QtWidgets.QVBoxLayout(activity_group)
        self.activity_table = QtWidgets.QTableWidget(0, 3)
        self.activity_table.setHorizontalHeaderLabels(["核素", "平均活度(Bq)", "峰数"])
        self.activity_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.activity_table.verticalHeader().setVisible(False)
        self.activity_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.activity_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.activity_table.setAlternatingRowColors(True)
        activity_layout.addWidget(self.activity_table)
        side_layout.addWidget(activity_group)

        side_layout.addStretch(1)

        content_layout.addWidget(side_panel, 1)
        main_layout.addLayout(content_layout, 1)
        self.statusBar().showMessage("教学/科研原型：结果仅供参考，不可替代经检定的测量软件")

        self.load_btn.clicked.connect(self.load_spectrum)
        self.background_btn.clicked.connect(self.import_background_spectrum)
        self.demo_btn.clicked.connect(self.load_demo_spectrum)
        self.calibrate_btn.clicked.connect(self.open_calibration_dialog)
        self.efficiency_btn.clicked.connect(self.open_efficiency_calibration_dialog)
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.manual_peaks_btn.clicked.connect(self.open_peak_selection_dialog)
        self.report_btn.clicked.connect(self.export_report)
        self.zoom_combo.currentIndexChanged.connect(self._apply_zoom_mode)
        self.reset_view_btn.clicked.connect(self._reset_view_range)
        self.smoothing_enable_check.toggled.connect(self.smoothing_window_spin.setEnabled)
        self.smoothing_enable_check.toggled.connect(lambda _: self._handle_smoothing_settings_changed())
        self.smoothing_window_spin.valueChanged.connect(lambda _: self._handle_smoothing_settings_changed())
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._update_cursor)

    def _setup_menus(self) -> None:
        configuration_menu = self.menuBar().addMenu("配置")
        self.analysis_settings_action = configuration_menu.addAction("分析参数…")
        configuration_menu.addSeparator()
        self.import_library_action = configuration_menu.addAction("导入核素库…")
        self.restore_library_action = configuration_menu.addAction("恢复内置教学库")
        self.certificate_library_action = configuration_menu.addAction("从证书生成核素库…")
        configuration_menu.addSeparator()
        self.import_background_action = configuration_menu.addAction("导入实测本底谱…")
        self.clear_background_action = configuration_menu.addAction("移除本底谱")
        self.background_scale_menu = configuration_menu.addMenu("本底归一方式")
        self.background_scale_group = QtWidgets.QActionGroup(self)
        for label, value in (("按活时间比", "live"), ("按实时间比", "real"), ("不归一 (1:1)", "none")):
            action = self.background_scale_menu.addAction(label)
            action.setCheckable(True)
            action.setData(value)
            action.setChecked(value == self.analysis_config.background_scale)
            self.background_scale_group.addAction(action)
        self.snip_after_background_action = configuration_menu.addAction("扣除本底后仍做 SNIP 估计")
        self.snip_after_background_action.setCheckable(True)
        self.snip_after_background_action.setChecked(True)

        calibration_menu = self.menuBar().addMenu("刻度")
        self.auto_calibration_action = calibration_menu.addAction("自动刻度（混合源）…")
        self.manual_calibration_action = calibration_menu.addAction("自定义刻度点…")
        self.import_calibration_action = calibration_menu.addAction("载入能量刻度…")
        self.export_calibration_action = calibration_menu.addAction("保存能量刻度…")

        help_menu = self.menuBar().addMenu("说明")
        self.limitations_action = help_menu.addAction("科学与使用限制")

        self.analysis_settings_action.triggered.connect(self.open_analysis_settings_dialog)
        self.import_library_action.triggered.connect(self.import_nuclide_library)
        self.restore_library_action.triggered.connect(self.restore_builtin_library)
        self.certificate_library_action.triggered.connect(self.import_certificate_library)
        self.import_background_action.triggered.connect(self.import_background_spectrum)
        self.clear_background_action.triggered.connect(self.clear_background_spectrum)
        self.snip_after_background_action.toggled.connect(self._handle_background_snip_toggled)
        self.background_scale_group.triggered.connect(self._handle_background_scale_changed)
        self.manual_calibration_action.triggered.connect(self.open_manual_calibration_dialog)
        self.auto_calibration_action.triggered.connect(self.run_auto_calibration)
        self.import_calibration_action.triggered.connect(self.import_energy_calibration)
        self.export_calibration_action.triggered.connect(self.export_energy_calibration)
        self.limitations_action.triggered.connect(self.show_limitations)

    # ---------------- 功能逻辑 ----------------
    def load_spectrum(self) -> None:
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择谱文件",
            BASE_DIR,
            SPECTRUM_FILE_FILTER,
        )
        if not filepath:
            return
        try:
            spectrum = load_spectrum_file(filepath)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.current_spectrum = spectrum
        self._reset_analysis_state()
        self._load_saved_efficiency_model()
        self.activity_status_message = "已导入"
        self._update_activity_summary_table()
        self._update_plot()
        self._update_info_label()
        self._update_peak_table()
        self.statusBar().showMessage(f"已加载 {os.path.basename(filepath)}")

    def import_background_spectrum(self) -> None:
        """Load a measured background and subtract it from the current sample."""

        if not self.current_spectrum:
            QtWidgets.QMessageBox.information(self, "提示", "请先导入样品谱，再导入本底谱。")
            return
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择实测本底谱",
            BASE_DIR,
            SPECTRUM_FILE_FILTER,
        )
        if not filepath:
            return
        try:
            background = load_spectrum_file(filepath)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(exc))
            return
        try:
            self.set_background(background)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "本底不兼容", str(exc))
            return
        self.statusBar().showMessage(
            f"已载入本底谱 {os.path.basename(filepath)}"
            f"（{self.background_info.describe_scale()} ×{self.background_info.scale:.3f}）"
        )

    def set_background(self, background: SpectrumRecord) -> None:
        """Attach a measured background spectrum and re-run the analysis."""

        if not self.current_spectrum:
            raise ValueError("请先导入样品谱，再导入本底谱。")
        _, info = subtract_background(
            self.current_spectrum,
            background,
            scale_mode=self.analysis_config.background_scale,
        )
        self.background_spectrum = background
        self.background_info = info
        self._reset_analysis_state()
        self._update_plot()
        self._update_info_label()
        self.activity_status_message = "本底谱已更新，请重新分析"
        self._update_activity_summary_table()
        self._update_peak_table()
        self.run_analysis()

    def clear_background_spectrum(self) -> None:
        if self.background_spectrum is None:
            return
        self.background_spectrum = None
        self.background_info = None
        self._reset_analysis_state()
        self._update_plot()
        self._update_info_label()
        self._update_peak_table()
        self.statusBar().showMessage("已移除本底谱，恢复 SNIP 自动本底估计")
        if self.current_spectrum:
            self.run_analysis()

    def _handle_background_snip_toggled(self, checked: bool) -> None:
        self.analysis_config = dataclasses.replace(self.analysis_config, snip_after_background=bool(checked))
        self.net_counts = None
        self.smoothed_net_counts = None
        if not self.current_spectrum:
            return
        self._update_plot()
        self._update_info_label()
        if self.peaks:
            self.run_analysis()
        self.statusBar().showMessage(
            "扣除本底后仍做 SNIP：%s" % ("开启" if checked else "关闭")
        )

    def _handle_background_scale_changed(self, action: QtWidgets.QAction) -> None:
        value = action.data()
        if value == self.analysis_config.background_scale:
            return
        self.analysis_config = dataclasses.replace(self.analysis_config, background_scale=value)
        self.net_counts = None
        self.smoothed_net_counts = None
        if not self.current_spectrum:
            return
        self._update_plot()
        self._update_info_label()
        if self.peaks:
            self.run_analysis()
        self.statusBar().showMessage(f"本底归一方式已切换为：{action.text()}")

    def _prepare_analysis_counts(self) -> Optional[np.ndarray]:
        """Counts that enter peak finding: sample minus normalized background."""

        if not self.current_spectrum:
            return None
        if self.background_spectrum is None:
            self.background_info = None
            self.analysis_counts = self.current_spectrum.counts
            return self.analysis_counts
        record, info = subtract_background(
            self.current_spectrum,
            self.background_spectrum,
            scale_mode=self.analysis_config.background_scale,
        )
        self.background_info = info
        self.analysis_counts = record.counts
        return self.analysis_counts

    def _background_display_counts(self) -> Optional[np.ndarray]:
        """Background counts scaled onto the sample grid, for the plot overlay."""

        if self.background_spectrum is None or not self.current_spectrum:
            return None
        background = self.background_spectrum
        scale = self.background_info.scale if self.background_info else 1.0
        sample_channels = self.current_spectrum.channels
        if len(background.channels) == len(sample_channels) and np.allclose(
            background.channels, sample_channels, rtol=0.0, atol=1e-6
        ):
            counts = background.counts
        else:
            counts = np.interp(sample_channels, background.channels, background.counts, left=0.0, right=0.0)
        return np.asarray(counts, dtype=float) * float(scale)

    def load_demo_spectrum(self) -> None:
        """Load a deterministic synthetic spectrum without using external data."""

        self.current_spectrum = synthetic_co60_record()
        self.calibration = CalibrationModel([DEMO_SLOPE_KEV_PER_CHANNEL, DEMO_OFFSET_KEV])
        self.calibration_source = "合成示例刻度"
        self.efficiency_model = EfficiencyModel()
        self.efficiency_points = []
        self._reset_analysis_state()
        self.activity_status_message = "未做探测器效率刻度，仅展示峰识别"
        self._update_activity_summary_table()
        self._update_plot()
        self._update_info_label()
        self._update_peak_table()
        self.run_analysis()
        self.statusBar().showMessage("已加载合成 Co-60 示例；该数据不是实测数据")

    def open_analysis_settings_dialog(self) -> None:
        dialog = AnalysisSettingsDialog(self, self.analysis_config)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.analysis_config = dialog.config()
        self._reset_analysis_state()
        if self.current_spectrum:
            self.run_analysis()
        self._update_info_label()
        self.statusBar().showMessage("分析参数已更新")

    def import_nuclide_library(self) -> None:
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入核素参考库",
            BASE_DIR,
            "核素库 (*.json *.JSON *.csv *.CSV);;所有文件 (*.*)",
        )
        if not filepath:
            return
        try:
            library = load_nuclide_library(filepath)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.nuclide_library = library
        self.library_source = os.path.basename(filepath)
        self.flat_library = flatten_library(library)
        self._rematch_current_peaks()
        self._update_info_label()
        self.statusBar().showMessage(
            f"已导入核素库 {self.library_source}：{len(library)} 种核素，{len(self.flat_library)} 条参考线"
        )

    def restore_builtin_library(self) -> None:
        self.nuclide_library = validate_nuclide_library(NNDC_LIBRARY)
        self.library_source = "内置教学库"
        self.flat_library = flatten_library(self.nuclide_library)
        self._rematch_current_peaks()
        self._update_info_label()
        self.statusBar().showMessage("已恢复内置教学核素库")

    def import_certificate_library(self) -> None:
        """Narrow the library down to what the standard source certificate covers."""

        dialog = CertificateLibraryDialog(self, self.nuclide_library)
        if dialog.exec_() != QtWidgets.QDialog.Accepted or dialog.result is None:
            return
        result = dialog.result
        self.nuclide_library = validate_nuclide_library(result.library)
        self.flat_library = flatten_library(self.nuclide_library)
        self.library_source = f"证书 {len(self.nuclide_library)} 种核素"
        ensure_directory(LIBRARY_DIR)
        library_path = os.path.join(LIBRARY_DIR, "certificate-library.json")
        try:
            with open(library_path, "w", encoding="utf-8") as handle:
                json.dump(self.nuclide_library, handle, ensure_ascii=False, indent=2)
            with open(os.path.join(LIBRARY_DIR, "certificate.txt"), "w", encoding="utf-8") as handle:
                handle.write(dialog.editor.toPlainText())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", f"核素库已应用，但保存到磁盘失败：{exc}")
        self._rematch_current_peaks()
        self._update_info_label()
        self.statusBar().showMessage(
            f"已按证书收缩核素库：{len(self.nuclide_library)} 种核素 / {len(self.flat_library)} 条参考线"
        )
        QtWidgets.QMessageBox.information(
            self,
            "证书核素库",
            result.summary_text() + f"\n\n已保存：{library_path}\n（可用“配置 → 恢复内置教学库”还原）",
        )

    def open_manual_calibration_dialog(self) -> None:
        dialog = ManualCalibrationDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        channels, energies, degree = dialog.values()
        try:
            model, residual = fit_energy_calibration(channels, energies, degree=degree)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "刻度失败", str(exc))
            return
        self.calibration = model
        self.calibration_source = "自定义刻度点"
        self._rematch_current_peaks()
        self._update_info_label()
        QtWidgets.QMessageBox.information(
            self,
            "能量刻度",
            f"已采用 {len(channels)} 个自定义刻度点。\n{model.describe()}\n均方根残差: {residual:.3f} keV",
        )

    def run_auto_calibration(self) -> None:
        """Enumerate mixed-source calibration hypotheses and keep the self-consistent one."""

        if not self.current_spectrum:
            QtWidgets.QMessageBox.information(self, "提示", "未导入谱文件。")
            return
        peaks = self.process_peaks()
        if len(peaks) < 4:
            QtWidgets.QMessageBox.warning(
                self,
                "峰数不足",
                f"只找到 {len(peaks)} 个峰。自动刻度需要对多条参考线投票，至少需要 4 个峰。",
            )
            return
        self.statusBar().showMessage("正在枚举候选刻度…")
        QtWidgets.QApplication.processEvents()
        result = auto_calibrate(
            peaks,
            self.nuclide_library,
            absolute_tolerance_kev=self.analysis_config.absolute_tolerance_kev,
            relative_tolerance=self.analysis_config.relative_tolerance,
        )
        self.statusBar().showMessage("自动刻度搜索完成")
        if result.model is None:
            QtWidgets.QMessageBox.warning(self, "自动刻度", result.note or "未能找到自洽的刻度。")
            return

        detail_lines = [
            f"{item.isotope:8s} {item.energy:9.3f} keV  通道 {item.channel:9.1f}  残差 {item.residual_kev:+0.3f} keV"
            for item in result.matched
        ]
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("自动能量刻度（混合源）")
        verdict = "通过" if result.credible else "未通过"
        box.setIcon(
            QtWidgets.QMessageBox.Information if result.credible else QtWidgets.QMessageBox.Warning
        )
        box.setText(f"可信度检查：{verdict}\n\n{result.summary_text()}")
        box.setDetailedText("命中的参考线：\n" + "\n".join(detail_lines))
        box.setInformativeText("是否把该刻度应用到当前谱？")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes if result.credible else QtWidgets.QMessageBox.No)
        box.button(QtWidgets.QMessageBox.Yes).setText("应用")
        box.button(QtWidgets.QMessageBox.No).setText("取消")
        details_button = None
        for candidate in box.buttons():
            if box.buttonRole(candidate) == QtWidgets.QMessageBox.ActionRole:
                details_button = candidate
                break
        if details_button is not None:
            details_button.setText("命中线详情")
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            self.statusBar().showMessage("已取消自动刻度")
            return

        self.calibration = result.model
        self.calibration_source = "自动刻度（混合源）"
        self.peaks = peaks
        self._rematch_current_peaks()
        self._update_info_label()
        self._update_plot()
        self.statusBar().showMessage(
            f"已应用自动刻度：{result.model.describe()}（命中 {len(result.matched)} 条参考线）"
        )

    def import_energy_calibration(self) -> None:
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "载入能量刻度",
            str(APP_DATA_DIR),
            "能量刻度 (*.json *.JSON);;所有文件 (*.*)",
        )
        if not filepath:
            return
        try:
            payload = json.loads(Path(filepath).read_text(encoding="utf-8-sig"))
            coefficients = payload.get("coefficients")
            if not isinstance(coefficients, list) or not 2 <= len(coefficients) <= 3:
                raise ValueError("系数必须是包含 2 或 3 个数值的数组。")
            values = [float(value) for value in coefficients]
            if not np.all(np.isfinite(values)):
                raise ValueError("刻度系数包含 NaN 或无穷值。")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "载入失败", f"能量刻度文件无效：{exc}")
            return
        self.calibration = CalibrationModel(values)
        self.calibration_source = f"刻度文件 {os.path.basename(filepath)}"
        self._rematch_current_peaks()
        self._update_info_label()
        self.statusBar().showMessage(f"已载入能量刻度 {os.path.basename(filepath)}")

    def export_energy_calibration(self) -> None:
        ensure_directory(str(APP_DATA_DIR))
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存能量刻度",
            str(APP_DATA_DIR / "energy_calibration.json"),
            "能量刻度 (*.json)",
        )
        if not filepath:
            return
        destination = Path(filepath)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        payload = {
            "schema_version": 1,
            "coefficients": self.calibration.coefficients,
            "description": self.calibration.describe(),
        }
        try:
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.statusBar().showMessage(f"已保存能量刻度 {destination.name}")

    def _rematch_current_peaks(self) -> None:
        if not self.peaks:
            return
        self._match_peaks(self.peaks)
        self._recalculate_activity_analysis()
        self._update_peak_table()

    def show_limitations(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "科学与使用限制",
            "本软件是教学/科研原型。核素候选和匹配分数必须人工复核；活度结果依赖具体探测器的效率刻度，"
            "且未自动处理衰变、几何、自吸收、符合相加或不确定度传播。\n\n"
            "不得将输出直接用于安全、医疗、监管、环境合规或计量证书。",
        )

    def _reset_analysis_state(self) -> None:
        self.net_counts = None
        self.analysis_counts = None
        self.smoothed_counts = None
        self.smoothed_net_counts = None
        self.peaks.clear()
        self.nuclide_summary = []
        self.activity_summary = []

    def _smoothing_enabled(self) -> bool:
        return bool(getattr(self, "smoothing_enable_check", None) and self.smoothing_enable_check.isChecked())

    def _smoothing_window(self) -> int:
        if not hasattr(self, "smoothing_window_spin"):
            return 7
        value = int(self.smoothing_window_spin.value())
        return value if value % 2 == 1 else value + 1

    def _handle_smoothing_settings_changed(self) -> None:
        self.smoothed_counts = None
        self.smoothed_net_counts = None
        if not self.current_spectrum:
            self._update_plot()
            return
        self.peaks.clear()
        self.nuclide_summary = []
        self.activity_summary = []
        self.activity_status_message = "平滑参数已更新，请重新分析"
        self._update_info_label()
        self._update_plot()
        self._update_peak_table()
        self.statusBar().showMessage("平滑辅助参数已更新")

    def _nearest_channel_index(self, channel: float) -> int:
        if not self.current_spectrum or len(self.current_spectrum.channels) == 0:
            return 0
        channels = self.current_spectrum.channels
        idx = int(np.searchsorted(channels, channel))
        if idx <= 0:
            return 0
        if idx >= len(channels):
            return len(channels) - 1
        left = idx - 1
        return idx if abs(channels[idx] - channel) < abs(channel - channels[left]) else left

    def _get_smoothed_counts(self) -> Optional[np.ndarray]:
        if not self.current_spectrum or not self._smoothing_enabled():
            return None
        if self.smoothed_counts is None:
            self.smoothed_counts = smooth_spectrum(self.current_spectrum.counts, self._smoothing_window())
        return self.smoothed_counts

    def _get_peak_detection_counts(self, net_counts: np.ndarray) -> np.ndarray:
        if not self._smoothing_enabled():
            return net_counts
        if self.smoothed_net_counts is None:
            self.smoothed_net_counts = smooth_spectrum(net_counts, self._smoothing_window())
        return self.smoothed_net_counts

    def ensure_net_counts(self) -> Optional[np.ndarray]:
        if not self.current_spectrum:
            QtWidgets.QMessageBox.warning(self, "提示", "未导入谱文件。")
            return None
        if self.net_counts is None:
            try:
                counts = self._prepare_analysis_counts()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "本底扣除失败", str(exc))
                return None
            if counts is None:
                return None
            if self.background_info is not None and not self.snip_after_background_action.isChecked():
                baseline = np.zeros_like(counts)
                self.background_info.continuum = "none"
                self.statusBar().showMessage("实测本底扣除完成（未做 SNIP 估计）")
            else:
                self.statusBar().showMessage("正在执行 SNIP 本底扣除…")
                baseline = snip_baseline(counts, iterations=self.analysis_config.snip_iterations)
                if self.background_info is not None:
                    self.background_info.continuum = "snip"
            self.net_counts = np.clip(counts - baseline, 0, None)
            self.smoothed_net_counts = None
            self.net_curve.show()
            self.net_curve.setData(self.current_spectrum.channels, self.net_counts)
            self.statusBar().showMessage("本底扣除完成")
        return self.net_counts

    def process_peaks(self) -> List[PeakResult]:
        net_counts = self.ensure_net_counts()
        if net_counts is None or not self.current_spectrum:
            return []
        candidates = detect_peaks_from_cwt(
            self._get_peak_detection_counts(net_counts),
            widths=self.analysis_config.wavelet_widths,
            min_distance=self.analysis_config.min_peak_distance,
            threshold_sigma=self.analysis_config.detection_threshold_sigma,
        )
        peaks = merge_close_peaks(
            refine_peaks(
                self.current_spectrum.channels,
                net_counts,
                self.current_spectrum.counts,
                candidates,
                min_snr=self.analysis_config.min_snr,
            )
        )
        return peaks

    def _match_peaks(self, peaks: List[PeakResult]) -> None:
        match_nuclides(
            peaks,
            self.calibration,
            self.flat_library,
            absolute_tolerance_kev=self.analysis_config.absolute_tolerance_kev,
            relative_tolerance=self.analysis_config.relative_tolerance,
            max_candidates=self.analysis_config.max_candidates,
            width_tolerance_factor=self.analysis_config.width_tolerance_factor,
            interference_width_factor=self.analysis_config.interference_width_factor,
        )
        self.nuclide_summary = summarize_nuclide_matches(
            peaks,
            self.nuclide_library,
            absolute_tolerance_kev=self.analysis_config.absolute_tolerance_kev,
            relative_tolerance=self.analysis_config.relative_tolerance,
            width_tolerance_factor=self.analysis_config.width_tolerance_factor,
        )
        if self.net_counts is not None and self.current_spectrum is not None:
            decompose_interference(
                peaks,
                self.current_spectrum.channels,
                self.net_counts,
                self.calibration,
            )

    def _close_peak_dialog(self) -> None:
        if self._peak_dialog:
            self._peak_dialog.blockSignals(True)
            self._peak_dialog.close()
            self._peak_dialog = None

    def _request_peak_selection(
        self,
        peaks: List[PeakResult],
        min_required: int,
        title: str,
        tip: str,
        on_accept: Callable[[List[PeakResult]], None],
        cancel_message: str = "已取消峰确认",
    ) -> None:
        self._close_peak_dialog()
        if not peaks:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可供选择的峰。")
            return
        dialog = PeakSelectionDialog(
            self,
            peaks,
            self.calibration,
            title=title,
            tip=tip,
            min_required=min_required,
            preview_callback=self._highlight_peak_preview,
        )
        dialog.accepted.connect(lambda d=dialog: self._handle_peak_selection_accept(d, on_accept))
        dialog.rejected.connect(lambda: self.statusBar().showMessage(cancel_message))
        dialog.finished.connect(lambda _: self._clear_peak_dialog_reference())
        dialog.show()
        self._peak_dialog = dialog

    def _handle_peak_selection_accept(
        self,
        dialog: PeakSelectionDialog,
        callback: Callable[[List[PeakResult]], None],
    ) -> None:
        selected = dialog.selected_peaks()
        callback(selected)

    def _clear_peak_dialog_reference(self) -> None:
        self._peak_dialog = None

    def _counts_value_at_channel(self, channel: float) -> float:
        if not self.current_spectrum:
            return 0.0
        idx = self._nearest_channel_index(channel)
        return float(self.current_spectrum.counts[idx])

    def _highlight_peak_preview(self, peak: PeakResult) -> None:
        if not self.current_spectrum:
            return
        y = self._counts_value_at_channel(peak.channel)
        self.v_line.setPos(peak.channel)
        self.h_line.setPos(y)
        self.cursor_label.setText(
            f"光标: 道址 {peak.channel:.1f} | 计数 {y:.1f} | 能量 {self.calibration.energy(peak.channel):.2f} keV"
        )

    def _handle_table_selection(self) -> None:
        if not self.peaks or not self.peaks_table.selectionModel():
            return
        indexes = self.peaks_table.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        if 0 <= row < len(self.peaks):
            self._highlight_peak_preview(self.peaks[row])
            self._focus_on_peak(self.peaks[row])

    def _clear_peak_annotations(self) -> None:
        for label in self.peak_labels:
            self.plot_widget.removeItem(label)
        self.peak_labels = []

    def _update_peak_annotations(self) -> None:
        self._clear_peak_annotations()
        if not self.current_spectrum or not self.peaks:
            return
        counts = self.current_spectrum.counts
        for peak in self.peaks:
            if not peak.is_ambiguous and not peak.nuclide_candidates:
                continue
            idx = self._nearest_channel_index(peak.channel)
            y = counts[idx]
            text_lines = [f"{peak.energy:.1f} keV"]
            if peak.is_ambiguous:
                text_lines.append(" + ".join(line.isotope for line in peak.interference[:2]) + " 双线")
            elif peak.nuclide_candidates:
                text_lines.append(peak.nuclide_candidates[0])
            if peak.confidence > 0:
                suffix = "†" if peak.is_ambiguous else ""
                text_lines.append(f"匹配分数 {peak.confidence * 100:.1f}%{suffix}")
            label = pg.TextItem(
                "\n".join(text_lines),
                color="#ff9f0a" if peak.is_ambiguous else ("#ffd60a" if peak.nuclide_candidates else "#d8dee9"),
            )
            label.setAnchor((0.5, 1.2))
            label.setPos(peak.channel, y)
            self.plot_widget.addItem(label)
            self.peak_labels.append(label)

    def _load_saved_efficiency_model(self) -> None:
        if not os.path.exists(EFFICIENCY_CACHE_PATH):
            return
        try:
            with open(EFFICIENCY_CACHE_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"效率刻度加载失败: {exc}")
            return
        coeffs = payload.get("coefficients")
        if not isinstance(coeffs, list) or not coeffs:
            return
        self.efficiency_model = EfficiencyModel([float(c) for c in coeffs])
        restored: List[EfficiencyPoint] = []
        for item in payload.get("points", []):
            try:
                restored.append(
                    EfficiencyPoint(
                        energy=float(item.get("energy", 0.0)),
                        efficiency=float(item.get("efficiency", 0.0)),
                        peak_channel=float(item.get("peak_channel", 0.0)),
                        peak_area=float(item.get("peak_area", 0.0)),
                        counts_per_second=float(item.get("counts_per_second", 0.0)),
                        intensity=float(item.get("intensity", 0.0)),
                        nuclide=str(item.get("nuclide", "")),
                    )
                )
            except Exception:
                continue
        if restored:
            self.efficiency_points = restored
            self.efficiency_model.set_energy_bounds([point.energy for point in restored])
        self.activity_status_message = payload.get("status_message", "已加载效率刻度")
        self.statusBar().showMessage("已加载效率刻度")
        self._update_activity_summary_table()

    def _save_efficiency_model(self) -> None:
        data = {
            "coefficients": self.efficiency_model.coefficients,
            "points": [
                {
                    "energy": point.energy,
                    "efficiency": point.efficiency,
                    "peak_channel": point.peak_channel,
                    "peak_area": point.peak_area,
                    "counts_per_second": point.counts_per_second,
                    "intensity": point.intensity,
                    "nuclide": point.nuclide,
                }
                for point in self.efficiency_points
            ],
            "status_message": self.activity_status_message,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            ensure_directory(str(APP_DATA_DIR))
            with open(EFFICIENCY_CACHE_PATH, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"效率刻度保存失败: {exc}")

    def _recalculate_activity_analysis(self) -> None:
        for peak in self.peaks:
            peak.activity_bq = None
        self.activity_summary = []
        self.activity_status_message = "等待效率刻度"
        if not self.peaks:
            self._update_activity_summary_table()
            return
        if not self.current_spectrum:
            self.activity_status_message = "未导入"
            self._update_activity_summary_table()
            return
        measurement_time = self.current_spectrum.live_time or self.current_spectrum.real_time
        if not measurement_time or measurement_time <= 0:
            self.activity_status_message = "缺少计时信息"
            self._update_activity_summary_table()
            return
        if not self.efficiency_model.is_valid():
            self.activity_status_message = "未完成效率刻度"
            self._update_activity_summary_table()
            return
        if not getattr(self.efficiency_model, "energy_bounds", None) and self.efficiency_points:
            self.efficiency_model.set_energy_bounds([point.energy for point in self.efficiency_points])
        measurement_time = float(measurement_time)
        efficiency_cache: Dict[float, float] = {}
        summary_map: Dict[str, List[float]] = {}
        self.activity_status_message = "暂无可用活度数据"
        for peak in self.peaks:
            cps = peak.area / measurement_time if measurement_time > 0 else 0.0
            if cps <= 0 or peak.energy <= 0 or not peak.primary_isotope or peak.reference_intensity <= 0:
                continue
            eff = efficiency_cache.get(peak.energy)
            if eff is None:
                eff = self.efficiency_model.efficiency(peak.energy)
                efficiency_cache[peak.energy] = eff
            if eff <= 0:
                continue
            activity = cps / (eff * peak.reference_intensity)
            if activity <= 0:
                continue
            peak.activity_bq = float(activity)
            summary_map.setdefault(peak.primary_isotope, []).append(peak.activity_bq)
        if not summary_map:
            self._update_activity_summary_table()
            return
        self.activity_summary = [
            ActivitySummary(isotope=isotope, activity_bq=float(np.mean(values)), peak_count=len(values))
            for isotope, values in summary_map.items()
        ]
        self.activity_summary.sort(key=lambda item: item.activity_bq, reverse=True)
        self.activity_status_message = "活度统计完成"
        self._update_activity_summary_table()

    def _update_view_limits(self) -> None:
        if not self.view_box:
            return
        vb = self.view_box
        if not self.current_spectrum or len(self.current_spectrum.channels) < 2:
            self._data_bounds = None
            vb.setAspectLocked(False)
            return
        channels = self.current_spectrum.channels
        counts = self.current_spectrum.counts
        x_min = float(channels[0])
        x_max = float(channels[-1])
        y_min = 0.0
        y_max = float(max(np.max(counts), 1.0) * 1.1 + 10.0)
        self._data_bounds = (x_min, x_max, y_min, y_max)
        vb.blockSignals(True)
        vb.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        vb.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0.02, update=True, disableAutoRange=True)
        vb.blockSignals(False)
        self._apply_zoom_mode()

    def _apply_zoom_mode(self) -> None:
        if not self.view_box:
            return
        mode = "locked"
        if hasattr(self, "zoom_combo"):
            data = self.zoom_combo.currentData()
            if data:
                mode = data
        self.view_box.set_mode(mode)
        if mode == "locked":
            self.view_box.setAspectLocked(True, ratio=self._current_ratio())

    def _current_ratio(self) -> float:
        if not self._data_bounds:
            return 1.0
        x0, x1, y0, y1 = self._data_bounds
        x_span = max(x1 - x0, 1.0)
        y_span = max(y1 - y0, 1.0)
        return x_span / y_span

    def _reset_view_range(self) -> None:
        if not self.view_box or not self._data_bounds:
            return
        x0, x1, y0, y1 = self._data_bounds
        self.view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.02, update=True)

    def _focus_on_peak(self, peak: PeakResult) -> None:
        if not self.view_box or not self._data_bounds:
            return
        x0, x1, y0, y1 = self._data_bounds
        span = max(x1 - x0, 1.0)
        window = max(span * 0.01, 20.0)
        left = max(x0, peak.channel - window)
        right = min(x1, peak.channel + window)
        y_value = self._counts_value_at_channel(peak.channel)
        top = min(y1, max(y_value * 1.4, y0 + 20))
        self.view_box.setRange(xRange=(left, right), yRange=(y0, top), padding=0.02, update=True)

    def _apply_manual_peaks_selection(self, selected: List[PeakResult]) -> None:
        if not selected:
            self.statusBar().showMessage("未选择任何峰。")
            return
        self._match_peaks(selected)
        self.peaks = selected
        self._recalculate_activity_analysis()
        self._update_peak_table()
        self.statusBar().showMessage(f"已确认 {len(selected)} 个峰")

    def _complete_calibration(self, isotope: str, peaks: List[PeakResult]) -> None:
        if len(peaks) < 2:
            QtWidgets.QMessageBox.warning(self, "提示", "至少选 2 个峰。")
            return
        reference_lines = sorted(self.nuclide_library.get(isotope, []), key=lambda item: -item.get("intensity", 0))
        if not reference_lines:
            QtWidgets.QMessageBox.warning(self, "提示", "核素缺少能量数据。")
            return
        usable = min(len(reference_lines), len(peaks))
        channels = np.array(sorted([peak.channel for peak in peaks], key=float)[:usable])
        energies = np.array(sorted([line["energy"] for line in reference_lines])[:usable])
        if usable < 2:
            QtWidgets.QMessageBox.warning(self, "提示", "匹配峰不足。")
            return
        try:
            linear_model, _ = fit_energy_calibration(channels, energies, degree=1)
            degree = 2 if usable >= 3 else 1
            self.calibration, residual = fit_energy_calibration(channels, energies, degree=degree)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "刻度失败", str(exc))
            return
        self._match_peaks(peaks)
        self.peaks = peaks
        self.calibration_source = f"标准源 {isotope}"
        self._recalculate_activity_analysis()
        self._update_peak_table()
        msg = (
            f"刻度完成。\n线性拟合: {linear_model.describe()}\n"
            f"采用拟合: {self.calibration.describe()}\n均方根残差: {residual:.3f} keV"
        )
        QtWidgets.QMessageBox.information(self, "刻度结果", msg)
        self.statusBar().showMessage("能量刻度已更新")

    def open_peak_selection_dialog(self) -> None:
        if not self.current_spectrum:
            QtWidgets.QMessageBox.warning(self, "提示", "未导入谱文件。")
            return
        peaks = self.process_peaks()
        if not peaks:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可确认的峰。")
            return
        self._request_peak_selection(
            peaks=peaks,
            min_required=1,
            title="人工确认峰",
            tip="勾选要保留的峰。",
            on_accept=self._apply_manual_peaks_selection,
            cancel_message="人工确认已取消",
        )

    def run_analysis(self) -> None:
        if not self.current_spectrum:
            QtWidgets.QMessageBox.information(self, "提示", "未导入谱文件。")
            return
        peaks = self.process_peaks()
        if not peaks:
            QtWidgets.QMessageBox.warning(self, "提示", "没有显著峰。")
            return
        self._match_peaks(peaks)
        self.peaks = peaks
        self._recalculate_activity_analysis()
        self._update_peak_table()
        self._update_info_label()
        self.statusBar().showMessage(f"已识别 {len(peaks)} 个峰")

    def open_calibration_dialog(self) -> None:
        if not self.current_spectrum:
            QtWidgets.QMessageBox.warning(self, "提示", "未导入谱文件。")
            return
        dialog = CalibrationDialog(self, list(self.nuclide_library.keys()))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.perform_calibration(dialog.selected_isotope())

    def perform_calibration(self, isotope: str) -> None:
        peaks = self.process_peaks()
        if len(peaks) < 2:
            QtWidgets.QMessageBox.warning(self, "提示", "可用于刻度的峰不足。")
            return
        self._request_peak_selection(
            peaks=peaks,
            min_required=2,
            title="选择用于刻度的峰",
            tip="选择标准源对应的峰，至少 2 个。",
            on_accept=lambda selected, iso=isotope: self._complete_calibration(iso, selected),
            cancel_message="能量刻度已取消",
        )

    def open_efficiency_calibration_dialog(self) -> None:
        if not self.current_spectrum:
            QtWidgets.QMessageBox.warning(self, "提示", "未导入谱文件。")
            return
        if self.calibration.is_default():
            QtWidgets.QMessageBox.warning(self, "提示", "先做能量刻度。")
            return
        dialog = EfficiencyCalibrationDialog(self, list(self.nuclide_library.keys()))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.perform_efficiency_calibration(
                dialog.selected_isotope(),
                dialog.activity_bq(),
                dialog.degree(),
            )

    def perform_efficiency_calibration(self, isotope: str, activity_bq: float, degree: int) -> None:
        if activity_bq <= 0:
            QtWidgets.QMessageBox.warning(self, "提示", "活度必须大于 0")
            return
        if not self.current_spectrum:
            QtWidgets.QMessageBox.warning(self, "提示", "未导入谱文件。")
            return
        measurement_time = self.current_spectrum.live_time or self.current_spectrum.real_time
        if not measurement_time or measurement_time <= 0:
            QtWidgets.QMessageBox.warning(self, "提示", "缺少计时信息。")
            return
        peaks = self.process_peaks()
        if not peaks:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可用峰。")
            return
        self._match_peaks(peaks)
        self.peaks = peaks
        self._update_peak_table()
        reference_lines = sorted(self.nuclide_library.get(isotope, []), key=lambda item: -item.get("intensity", 0))
        if not reference_lines:
            QtWidgets.QMessageBox.warning(self, "提示", "标准源缺少能量线数据。")
            return

        measurement_time = float(measurement_time)
        used_ids = set()
        raw_points: List[EfficiencyPoint] = []

        for line in reference_lines:
            energy = float(line.get("energy", 0.0))
            intensity = float(line.get("intensity", 0.0))
            if energy <= 0 or intensity <= 0:
                continue
            tolerance = max(0.5, energy * 0.003)
            candidates = []
            for peak in peaks:
                if id(peak) in used_ids or peak.energy <= 0:
                    continue
                diff = abs(peak.energy - energy)
                if diff <= tolerance:
                    candidates.append((diff, peak))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            best_peak = candidates[0][1]
            used_ids.add(id(best_peak))
            if best_peak.area <= 0:
                continue
            cps = best_peak.area / measurement_time
            if cps <= 0:
                continue
            efficiency = cps / (activity_bq * intensity)
            if efficiency <= 0:
                continue
            raw_points.append(
                EfficiencyPoint(
                    energy=energy,
                    efficiency=efficiency,
                    peak_channel=best_peak.channel,
                    peak_area=best_peak.area,
                    counts_per_second=cps,
                    intensity=intensity,
                    nuclide=isotope,
                )
            )

        valid_points = [point for point in raw_points if point.energy > 0 and point.efficiency > 0]
        if len(valid_points) < degree + 1:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                f"效率点不足: 需要 {degree + 1} 个，当前 {len(valid_points)} 个。",
            )
            return

        log_energy = np.log([point.energy for point in valid_points])
        log_eff = np.log([point.efficiency for point in valid_points])
        try:
            coeffs = np.polyfit(log_energy, log_eff, degree)
        except Exception as exc:  # pragma: no cover - 保守处理数值问题
            QtWidgets.QMessageBox.warning(self, "提示", f"效率刻度拟合失败：{exc}")
            return

        coeff_list = [float(c) for c in coeffs]
        energy_values = [point.energy for point in valid_points]
        self.efficiency_model = EfficiencyModel(coeff_list)
        self.efficiency_model.set_energy_bounds(energy_values)
        self.efficiency_points = valid_points
        self._recalculate_activity_analysis()
        self._update_peak_table()
        self._save_efficiency_model()

        preview = "\n".join(
            f"  {point.energy:.1f} keV -> ε={point.efficiency:.4e}"
            for point in valid_points[: min(5, len(valid_points))]
        )
        message = (
            f"效率刻度完成，{len(valid_points)} 个点，阶数 {degree}。\n"
            f"{self.efficiency_model.describe()}\n"
            f"效率点:\n{preview}"
        )
        QtWidgets.QMessageBox.information(self, "效率刻度", message)
        self.statusBar().showMessage(f"效率刻度完成，{len(valid_points)} 个点")

    def _background_report_text(self) -> str:
        info = self.background_info
        if self.background_spectrum is None or info is None:
            return "未使用（连续本底由 SNIP 自动估计）"
        parts = [
            os.path.basename(info.filepath),
            info.describe_scale(),
            f"×{info.scale:.4f}",
            "道址已重采样" if info.resampled else "道址一致",
            f"截断通道 {info.clipped_channels} ({info.clipped_fraction * 100:.1f}%)",
            "扣除后仍做 SNIP 估计" if info.continuum == "snip" else "扣除后不再做 SNIP 估计",
        ]
        return " | ".join(parts)

    def export_report(self) -> None:
        if not self.current_spectrum or not self.peaks:
            QtWidgets.QMessageBox.warning(self, "提示", "请先分析并确认峰。")
            return
        ensure_directory(REPORT_DIR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出分析报告",
            os.path.join(REPORT_DIR, f"report-{timestamp}.txt"),
            "文本报告 (*.txt)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".txt"):
            filename += ".txt"
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(self.build_report_text())
        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出: {filename}")
        self.statusBar().showMessage(f"已导出 {filename}")

    def build_report_text(self) -> str:
        """Render the plain-text report without touching the file system."""

        if not self.current_spectrum:
            return ""
        lines = [
            "谱线分析报告",
            "=" * 40,
            f"导出时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"数据文件: {self.current_spectrum.filepath}",
            f"实时间/活时间: {self.current_spectrum.real_time} / {self.current_spectrum.live_time}",
            f"能量刻度: {self.calibration.describe()}（{self.calibration_source}）",
            f"效率刻度: {self.efficiency_model.describe()}",
            f"核素库: {self.library_source} ({len(self.nuclide_library)} 种核素 / {len(self.flat_library)} 条参考线)",
            f"分析参数: {json.dumps(self.analysis_config.as_dict(), ensure_ascii=False)}",
            f"平滑辅助: {'开启' if self._smoothing_enabled() else '关闭'}"
            + (f" (窗口 {self._smoothing_window()})" if self._smoothing_enabled() else ""),
            f"实测本底: {self._background_report_text()}",
            "",
            "峰值信息:",
        ]
        if self.current_spectrum.warnings:
            lines.insert(8, "导入警告: " + "；".join(self.current_spectrum.warnings))
        for idx, peak in enumerate(self.peaks, 1):
            candidate_text = (
                " + ".join(f"{line.isotope}({line.energy:.1f})" for line in peak.interference)
                if peak.is_ambiguous
                else (", ".join(peak.nuclide_candidates) or "—")
            )
            confidence_text = f"匹配分数 {peak.confidence * 100:.1f}%"
            if peak.is_ambiguous:
                confidence_text = "匹配分数 —（双线不可分辨）"
            area_text = f"{peak.area:.1f}"
            if peak.area_uncertainty > 0:
                area_text += f" ± {peak.area_uncertainty:.1f}"
            lines.append(
                f"[{idx}] 通道 {peak.channel:.1f}	能量 {peak.energy:.2f} keV	面积 {area_text}	"
                f"FWHM {peak.fwhm:.2f} ch / {peak.fwhm_kev:.2f} keV	候选 {candidate_text}	"
                f"{confidence_text}	活度 {format_activity(peak.activity_bq)}"
            )
        ambiguous_peaks = [peak for peak in self.peaks if peak.is_ambiguous]
        unfitted = [peak for peak in self.peaks if peak.area_uncertainty <= 0]
        if unfitted:
            lines.append("")
            lines.append(
                f"  注：{len(unfitted)} 个峰未通过高斯拟合一致性检查，面积按窗口求和给出、未估计不确定度。"
            )
        if ambiguous_peaks:
            lines.append("")
            lines.append("双线 / 干扰说明（标准为“库内两条参考线间距小于本峰 FWHM，本次未指定主核素”）:")
            for peak in ambiguous_peaks:
                detail = " / ".join(
                    f"{line.isotope} {line.energy:.3f} keV（Δ={line.delta_kev:+.3f}，等活度发射概率份额 {line.weight * 100:.0f}%）"
                    for line in peak.interference
                )
                lines.append(
                    f"  {peak.energy:.2f} keV（FWHM {peak.fwhm_kev:.2f} keV）：{detail}"
                )
                if peak.components:
                    verdict = "可采信" if peak.decomposition_resolved else "不采信"
                    lines.append(f"    → 固定能量多峰拟合（{verdict}）：")
                    for component in peak.components:
                        lines.append(f"      {component.as_text()}")
                    if peak.decomposition_note:
                        lines.append(f"      说明：{peak.decomposition_note}")
            lines.append("  → 该峰的活度与核素归属需用其他谱线或更高分辨率复测来确认，软件不做强行指派。")
        if self.nuclide_summary:
            lines.append("")
            lines.append("整谱核素候选（启发式）:")
            for item in self.nuclide_summary[:8]:
                ambiguous_note = f"（含 {item.ambiguous_lines} 条双线）" if item.ambiguous_lines else ""
                lines.append(
                    f"  {item.isotope} | 匹配线 {item.matched_lines}/{item.total_lines}{ambiguous_note} | "
                    f"强度覆盖 {item.intensity_coverage * 100:.1f}% | 谱级分数 {item.match_score * 100:.1f}%"
                )
        if self.efficiency_points:
            lines.append("")
            lines.append("效率刻度点:")
            for point in self.efficiency_points:
                lines.append(
                    f"  {point.energy:.1f} keV | ε={point.efficiency:.4e} | cps={point.counts_per_second:.2f} | 面积 {point.peak_area:.1f}"
                )
        if self.activity_summary:
            lines.append("")
            lines.append("核素活度汇总:")
            for item in self.activity_summary:
                lines.append(
                    f"  {item.isotope} | 活度 {format_activity(item.activity_bq)} | 峰数量 {item.peak_count}"
                )
        lines.extend(
            [
                "",
                "限制声明:",
                "  本软件为教学/科研原型，核素候选和匹配分数仅供人工复核。",
                "  实测本底按时间归一后逐道相减，负值截断到 0；该截断与峰面积的不确定度均未传播。",
                "  双线（†）峰不指定主核素、不计算活度，需用其他谱线或更高分辨率复测确认。",
                "  活度为基于输入效率曲线的估算值，未自动校正衰变、几何、自吸收或符合相加效应。",
                "  不得将输出用于安全、医疗、监管或环境合规决策。",
            ]
        )
        return "\n".join(lines)

    # ---------------- UI 辅助 ----------------
    def _update_plot(self) -> None:
        if not self.current_spectrum:
            self.raw_curve.setData([], [])
            self.smooth_curve.setData([], [])
            self.net_curve.setData([], [])
            self.background_curve.setData([], [])
            self.smooth_curve.hide()
            self.net_curve.hide()
            self.background_curve.hide()
            if self.view_box:
                self.view_box.setAspectLocked(False)
                self.view_box.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            return
        self.raw_curve.setData(self.current_spectrum.channels, self.current_spectrum.counts)
        smoothed = self._get_smoothed_counts()
        if smoothed is not None:
            self.smooth_curve.show()
            self.smooth_curve.setData(self.current_spectrum.channels, smoothed)
        else:
            self.smooth_curve.hide()
            self.smooth_curve.setData([], [])
        if self.net_counts is not None:
            self.net_curve.show()
            self.net_curve.setData(self.current_spectrum.channels, self.net_counts)
        else:
            self.net_curve.hide()
            self.net_curve.setData([], [])
        background_display = self._background_display_counts()
        if background_display is not None:
            self.background_curve.show()
            self.background_curve.setData(self.current_spectrum.channels, background_display)
        else:
            self.background_curve.hide()
            self.background_curve.setData([], [])
        self._update_view_limits()

    def _update_info_label(self) -> None:
        if not self.current_spectrum:
            self.info_label.setText("当前未加载谱文件")
            return
        name = os.path.basename(self.current_spectrum.filepath)
        rt = self.current_spectrum.real_time or "未知"
        lt = self.current_spectrum.live_time or "未知"
        smoothing_text = "开启" if self._smoothing_enabled() else "关闭"
        if self._smoothing_enabled():
            smoothing_text += f" (窗口 {self._smoothing_window()})"
        warning_text = f"\n导入警告: {self.current_spectrum.warnings[0]}" if self.current_spectrum.warnings else ""
        background_text = "无（使用 SNIP 自动估计）"
        if self.background_spectrum is not None and self.background_info is not None:
            background_text = (
                f"已扣除 {os.path.basename(self.background_spectrum.filepath)}"
                f"（{self.background_info.describe_scale()} ×{self.background_info.scale:.3f}）"
            )
        self.info_label.setText(
            f"当前文件: {name}\n实时间/活时间: {rt} / {lt}\n"
            f"刻度: {self.calibration_source}｜{self.calibration.describe()}\n"
            f"本底: {background_text}\n"
            f"核素库: {self.library_source}｜平滑辅助: {smoothing_text}{warning_text}"
        )

    def _update_peak_table(self) -> None:
        self.peaks_table.setRowCount(0)
        for peak in self.peaks:
            row = self.peaks_table.rowCount()
            self.peaks_table.insertRow(row)
            for col, value in enumerate(peak.as_row()):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                if col == 2 and peak.area_uncertainty > 0 and peak.area > 0:
                    item.setToolTip(
                        f"{peak.area:.1f} ± {peak.area_uncertainty:.1f} "
                        f"（{peak.area_uncertainty / peak.area * 100:.1f}%，高斯拟合统计不确定度）"
                    )
                if col == 4 and peak.components:
                    detail = "；".join(component.as_text() for component in peak.components)
                    verdict = "可采信" if peak.decomposition_resolved else "不采信"
                    item.setToolTip(f"{value}\n固定能量多峰拟合（{verdict}）：{detail}\n{peak.decomposition_note}")
                self.peaks_table.setItem(row, col, item)
        self._update_peak_annotations()
        self._update_identification_summary_table()
        self._update_activity_summary_table()

    def _update_identification_summary_table(self) -> None:
        if not hasattr(self, "identification_table"):
            return
        self.identification_table.setRowCount(0)
        for summary in self.nuclide_summary[:8]:
            row = self.identification_table.rowCount()
            self.identification_table.insertRow(row)
            values = (
                summary.isotope,
                f"{summary.matched_lines}/{summary.total_lines}"
                + (f" (含 {summary.ambiguous_lines}†)" if summary.ambiguous_lines else ""),
                f"{summary.intensity_coverage * 100:.1f}%",
                f"{summary.match_score * 100:.1f}%",
            )
            for column, value in enumerate(values):
                self.identification_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _update_activity_summary_table(self) -> None:
        if not hasattr(self, "activity_table"):
            return
        self.activity_table.setRowCount(0)
        if not self.activity_summary:
            item = QtWidgets.QTableWidgetItem(self.activity_status_message)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.activity_table.insertRow(0)
            self.activity_table.setItem(0, 0, item)
            self.activity_table.setSpan(0, 0, 1, self.activity_table.columnCount())
            for col in range(1, self.activity_table.columnCount()):
                placeholder = QtWidgets.QTableWidgetItem("")
                placeholder.setFlags(QtCore.Qt.ItemIsEnabled)
                self.activity_table.setItem(0, col, placeholder)
            return
        for summary in self.activity_summary:
            row = self.activity_table.rowCount()
            self.activity_table.insertRow(row)
            self.activity_table.setItem(row, 0, QtWidgets.QTableWidgetItem(summary.isotope))
            self.activity_table.setItem(row, 1, QtWidgets.QTableWidgetItem(format_activity(summary.activity_bq)))
            self.activity_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(summary.peak_count)))

    def _update_cursor(self, event: object) -> None:
        if not self.current_spectrum:
            return
        pos = event[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        vb = self.plot_widget.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        self.v_line.setPos(x)
        self.h_line.setPos(y)
        energy = self.calibration.energy(x)
        self.cursor_label.setText(f"光标: 道址 {x:.1f} | 计数 {y:.1f} | 能量 {energy:.2f} keV")


def main() -> None:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0f111a"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#f2f2f7"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#1c1e2a"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#2a2d3c"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#f2f2f7"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#1f2233"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#f2f2f7"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#3d7eff"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#000000"))
    app.setPalette(palette)
    window = SpectrumAnalyzerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
