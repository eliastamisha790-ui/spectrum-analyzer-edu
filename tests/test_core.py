from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectrum_analyzer.core import (
    AnalysisConfig,
    CalibrationModel,
    NNDC_LIBRARY,
    PeakResult,
    SpectrumRecord,
    analyze_spectrum,
    fit_energy_calibration,
    load_nuclide_library,
    load_spectrum_file,
    summarize_nuclide_matches,
)
from spectrum_analyzer.demo import (
    DEMO_OFFSET_KEV,
    DEMO_SLOPE_KEV_PER_CHANNEL,
    synthetic_co60_record,
    write_spe,
)


class SpectrumIoTests(unittest.TestCase):
    def test_spe_range_and_measurement_times(self) -> None:
        record = SpectrumRecord(
            filepath="fixture.spe",
            channels=np.arange(10, 15, dtype=float),
            counts=np.array([1, 2, 3, 4, 5], dtype=float),
            live_time=9.5,
            real_time=10.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_spe(record, Path(temp_dir) / "fixture.spe")
            parsed = load_spectrum_file(str(path))
        np.testing.assert_array_equal(parsed.channels, record.channels)
        np.testing.assert_array_equal(parsed.counts, record.counts)
        self.assertEqual(parsed.live_time, 9.5)
        self.assertEqual(parsed.real_time, 10.0)

    def test_csv_channel_count_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.csv"
            path.write_text("channel,count\n0,3\n1,4\n2,5\n3,6\n", encoding="utf-8")
            parsed = load_spectrum_file(str(path))
        np.testing.assert_array_equal(parsed.channels, [0, 1, 2, 3])
        np.testing.assert_array_equal(parsed.counts, [3, 4, 5, 6])

    def test_spe_exclusive_end_export_is_accepted_with_warning(self) -> None:
        content = "$DATA:\n0 4\n1\n2\n3\n4\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "exclusive-end.spe"
            path.write_text(content, encoding="utf-8")
            parsed = load_spectrum_file(str(path))
        np.testing.assert_array_equal(parsed.channels, [0, 1, 2, 3])
        self.assertEqual(len(parsed.warnings), 1)
        self.assertIn("上界", parsed.warnings[0])

    def test_custom_nuclide_library_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            json_path = folder / "library.json"
            json_path.write_text(
                '{"demo-x": [{"energy_kev": 100.0, "emission_probability": 0.8}]}',
                encoding="utf-8",
            )
            csv_path = folder / "library.csv"
            csv_path.write_text("nuclide,energy_kev,intensity\ndemo-y,200.0,0.5\n", encoding="utf-8")
            json_library = load_nuclide_library(str(json_path))
            csv_library = load_nuclide_library(str(csv_path))
        self.assertEqual(json_library["demo-x"][0]["energy"], 100.0)
        self.assertEqual(csv_library["demo-y"][0]["intensity"], 0.5)


class AnalysisTests(unittest.TestCase):
    def test_linear_energy_calibration(self) -> None:
        model, rmse = fit_energy_calibration([100, 200, 300], [50, 100, 150], degree=1)
        self.assertAlmostEqual(model.energy(250), 125.0, places=8)
        self.assertLess(rmse, 1e-10)

    def test_synthetic_co60_peaks_are_detected_and_matched(self) -> None:
        record = synthetic_co60_record()
        calibration = CalibrationModel([DEMO_SLOPE_KEV_PER_CHANNEL, DEMO_OFFSET_KEV])
        result = analyze_spectrum(record, calibration)
        co60 = [peak for peak in result.peaks if peak.primary_isotope == "co-60"]
        self.assertGreaterEqual(len(co60), 2)
        energies = sorted(peak.energy for peak in co60)
        self.assertAlmostEqual(energies[0], 1173.237, delta=3.0)
        self.assertAlmostEqual(energies[-1], 1332.501, delta=3.0)

    def test_negative_counts_are_rejected(self) -> None:
        record = SpectrumRecord(
            filepath="bad.csv",
            channels=np.arange(4, dtype=float),
            counts=np.array([1, 2, -1, 4], dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            analyze_spectrum(record)

    def test_custom_library_and_analysis_config_are_used(self) -> None:
        record = synthetic_co60_record()
        calibration = CalibrationModel([DEMO_SLOPE_KEV_PER_CHANNEL, DEMO_OFFSET_KEV])
        library = {
            "custom-source": [
                {"energy": 1173.237, "intensity": 0.999},
                {"energy": 1332.501, "intensity": 0.999},
            ]
        }
        config = AnalysisConfig(max_candidates=1, relative_tolerance=0.001)
        result = analyze_spectrum(record, calibration, library=library, config=config)
        matched = [peak for peak in result.peaks if peak.primary_isotope == "custom-source"]
        self.assertEqual(len(matched), 2)
        self.assertTrue(all(len(peak.nuclide_candidates) == 1 for peak in matched))

    def test_spectrum_level_summary_rewards_multiple_reference_lines(self) -> None:
        peaks = [
            PeakResult(channel=energy, energy=energy, fwhm=2.0, area=100.0, snr=10.0)
            for energy in (121.782, 344.278, 778.903, 964.055, 1112.116, 1408.006)
        ]
        summary = summarize_nuclide_matches(peaks, NNDC_LIBRARY)
        self.assertEqual(summary[0].isotope, "eu-152")
        self.assertGreaterEqual(summary[0].matched_lines, 6)


if __name__ == "__main__":
    unittest.main()
