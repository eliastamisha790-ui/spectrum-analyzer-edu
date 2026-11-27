"""Render the synthetic 186 keV doublet example to docs/screenshot-doublet.png."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from spectrum_analyzer.app import SpectrumAnalyzerApp
from spectrum_analyzer.core import CalibrationModel, load_spectrum_file
from spectrum_analyzer.demo import DOUBLET_OFFSET_KEV, DOUBLET_SLOPE_KEV_PER_CHANNEL


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SpectrumAnalyzerApp()
    window.resize(1440, 900)
    source = Path(__file__).resolve().parents[1] / "examples" / "synthetic_doublet.spe"
    window.current_spectrum = load_spectrum_file(str(source))
    window.calibration = CalibrationModel([DOUBLET_SLOPE_KEV_PER_CHANNEL, DOUBLET_OFFSET_KEV])
    window._reset_analysis_state()
    window._update_plot()
    window._update_info_label()
    window.run_analysis()
    window.show()
    app.processEvents()
    # 放大到 186 keV 附近，让双线标注成为画面主体
    window.view_box.setRange(xRange=(1215.0, 1500.0), yRange=(0.0, 620.0), padding=0.0, update=True)
    app.processEvents()

    destination = Path(__file__).resolve().parents[1] / "docs" / "screenshot-doublet.png"
    image = window.grab()
    if not image.save(str(destination), "PNG"):
        raise RuntimeError(f"无法保存截图：{destination}")
    print(destination)

    window.close()
    QtCore.QTimer.singleShot(0, app.quit)
    app.processEvents()


if __name__ == "__main__":
    main()
