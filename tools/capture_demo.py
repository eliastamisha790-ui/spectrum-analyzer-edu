"""Render the synthetic demo window to the README screenshot."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from spectrum_analyzer.app import SpectrumAnalyzerApp


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SpectrumAnalyzerApp()
    window.resize(1440, 900)
    window.load_demo_spectrum()
    window.show()
    app.processEvents()

    destination = Path(__file__).resolve().parents[1] / "docs" / "screenshot.png"
    image = window.grab()
    if not image.save(str(destination), "PNG"):
        raise RuntimeError(f"无法保存截图：{destination}")
    print(destination)

    window.close()
    QtCore.QTimer.singleShot(0, app.quit)
    app.processEvents()


if __name__ == "__main__":
    main()
