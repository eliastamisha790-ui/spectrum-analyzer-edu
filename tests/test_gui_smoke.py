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

    def test_measured_background_is_reflected_in_ui(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import numpy as np
        from PyQt5 import QtWidgets

        from spectrum_analyzer.app import SpectrumAnalyzerApp
        from spectrum_analyzer.core import SpectrumRecord

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = SpectrumAnalyzerApp()
        window.load_demo_spectrum()
        self.assertIsNone(window.background_info)

        record = window.current_spectrum
        background = SpectrumRecord(
            filepath="synthetic-lab-background.spe",
            channels=record.channels,
            counts=np.full_like(record.counts, 3.0),
            live_time=record.live_time,
            real_time=record.real_time,
        )
        window.set_background(background)

        self.assertIsNotNone(window.background_info)
        self.assertAlmostEqual(window.background_info.scale, 1.0, places=6)
        self.assertIn("已扣除", window.info_label.text())
        self.assertGreaterEqual(window.peaks_table.rowCount(), 2)

        window.clear_background_spectrum()
        self.assertIsNone(window.background_info)
        self.assertIn("无（使用 SNIP 自动估计）", window.info_label.text())

        window.close()
        app.processEvents()

    def test_unresolved_doublet_is_marked_in_table(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import numpy as np
        from PyQt5 import QtWidgets

        from spectrum_analyzer.app import SpectrumAnalyzerApp
        from spectrum_analyzer.core import CalibrationModel, SpectrumRecord

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        channels = np.arange(2048, dtype=float)
        counts = 5.0 + 800.0 * np.exp(-0.5 * ((channels - 186.0) / 3.0) ** 2)
        window = SpectrumAnalyzerApp()
        window.current_spectrum = SpectrumRecord(
            filepath="doublet.spe",
            channels=channels,
            counts=counts,
            live_time=100.0,
        )
        window.calibration = CalibrationModel([1.0, 0.0])
        window._reset_analysis_state()
        window._update_info_label()
        window.run_analysis()

        self.assertEqual(window.peaks_table.rowCount(), 1)
        candidate_cell = window.peaks_table.item(0, 4).text()
        self.assertIn("†", candidate_cell)
        self.assertIn("u-235", candidate_cell)
        self.assertIn("ra-226", candidate_cell)
        self.assertEqual(window.peaks[0].primary_isotope, "")
        self.assertIsNone(window.peaks[0].activity_bq)

        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
