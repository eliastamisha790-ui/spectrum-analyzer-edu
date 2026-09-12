from __future__ import annotations

import importlib.util
import os
import unittest


@unittest.skipUnless(importlib.util.find_spec("PyQt5"), "PyQt5 optional dependency is not installed")
class GuiSmokeTests(unittest.TestCase):
    def test_demo_can_load_and_analyze_offscreen(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from PyQt5 import QtWidgets

        from spectrum_analyzer.app import SpectrumAnalyzerApp

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = SpectrumAnalyzerApp()
        window.load_demo_spectrum()

        self.assertEqual(window.current_spectrum.filepath, "synthetic-co60-demo.spe")
        self.assertEqual([peak.primary_isotope for peak in window.peaks], ["co-60", "co-60"])
        self.assertEqual(window.peaks_table.rowCount(), 2)
        self.assertEqual([action.text() for action in window.menuBar().actions()], ["配置", "刻度", "说明"])
        self.assertEqual(window.nuclide_summary[0].isotope, "co-60")
        self.assertEqual(window.identification_table.rowCount(), 1)

        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
