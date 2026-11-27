from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectrum_analyzer.core import (
    AnalysisConfig,
    CalibrationModel,
    InterferenceLine,
    NNDC_LIBRARY,
    PeakResult,
    SpectrumRecord,
    analyze_spectrum,
    decompose_interference,
    fit_energy_calibration,
    fit_peak_shape,
    flatten_library,
    load_nuclide_library,
    load_spectrum_file,
    match_nuclides,
    merge_close_peaks,
    subtract_background,
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


class InterferenceTests(unittest.TestCase):
    def test_builtin_library_covers_natural_decay_chains(self) -> None:
        for isotope in (
            "u-235",
            "ra-226",
            "pb-210",
            "th-234",
            "pb-214",
            "bi-214",
            "ac-228",
            "pb-212",
            "bi-212",
            "tl-208",
            "k-40",
        ):
            self.assertIn(isotope, NNDC_LIBRARY, f"内置库缺少 {isotope}")

    def test_u235_ra226_doublet_is_flagged_without_primary_isotope(self) -> None:
        # 185.97 keV is the unresolved u-235 (185.715) / ra-226 (186.211) blend.
        peaks = [PeakResult(channel=1355.94, fwhm=10.44, area=4051.0, snr=19.0)]
        match_nuclides(
            peaks,
            CalibrationModel([0.137145, 0.0133]),
            flatten_library(NNDC_LIBRARY),
        )
        peak = peaks[0]
        self.assertTrue(peak.is_ambiguous)
        self.assertEqual(peak.primary_isotope, "")
        self.assertEqual(peak.reference_intensity, 0.0)
        self.assertEqual({line.isotope for line in peak.interference}, {"u-235", "ra-226"})
        self.assertAlmostEqual(sum(line.weight for line in peak.interference), 1.0, places=6)
        self.assertIn("†", peak.as_row()[4])

    def test_single_line_peak_keeps_primary_isotope(self) -> None:
        peaks = [PeakResult(channel=4442.93, fwhm=10.79, area=23964.0, snr=71.0)]
        match_nuclides(peaks, CalibrationModel([0.137145, 0.0133]), flatten_library(NNDC_LIBRARY))
        peak = peaks[0]
        self.assertAlmostEqual(peak.energy, 609.31, delta=0.05)
        self.assertFalse(peak.is_ambiguous)
        self.assertEqual(peak.primary_isotope, "bi-214")
        self.assertGreater(peak.confidence, 0.5)
        self.assertIn("bi-214", peak.as_row()[4])

    def test_tolerance_follows_peak_width_on_low_energy_lines(self) -> None:
        # 60.4 keV is 0.86 keV from am-241 59.541: outside the fixed 0.5 keV
        # tolerance but inside half of a 2 keV FWHM.
        peak = PeakResult(channel=60.4, fwhm=2.0, area=100.0, snr=10.0)
        match_nuclides([peak], CalibrationModel([1.0, 0.0]), flatten_library(NNDC_LIBRARY))
        self.assertEqual(peak.primary_isotope, "am-241")

    def test_ambiguous_peaks_only_get_half_line_credit_in_summary(self) -> None:
        peaks = [PeakResult(channel=185.97, fwhm=1.43, area=4051.0, snr=19.0)]
        match_nuclides(peaks, CalibrationModel([1.0, 0.0]), flatten_library(NNDC_LIBRARY))
        self.assertTrue(peaks[0].is_ambiguous)
        summary = summarize_nuclide_matches(peaks, NNDC_LIBRARY)
        by_isotope = {item.isotope: item for item in summary}
        self.assertIn("ra-226", by_isotope)
        self.assertEqual(by_isotope["ra-226"].ambiguous_lines, 1)
        self.assertLess(by_isotope["ra-226"].intensity_coverage, 0.6)


class BackgroundTests(unittest.TestCase):
    def test_live_time_scaling_removes_matched_counts(self) -> None:
        sample = SpectrumRecord(
            filepath="sample.spe",
            channels=np.arange(10, dtype=float),
            counts=np.full(10, 20.0),
            live_time=100.0,
            real_time=110.0,
        )
        background = SpectrumRecord(
            filepath="background.spe",
            channels=np.arange(10, dtype=float),
            counts=np.full(10, 10.0),
            live_time=50.0,
            real_time=60.0,
        )
        net, info = subtract_background(sample, background)
        self.assertAlmostEqual(info.scale, 2.0)
        np.testing.assert_allclose(net.counts, np.zeros(10))
        self.assertEqual(info.clipped_channels, 0)
        self.assertFalse(info.resampled)

    def test_negative_channels_are_clipped_and_reported(self) -> None:
        sample = SpectrumRecord(
            filepath="sample.spe",
            channels=np.arange(8, dtype=float),
            counts=np.array([5, 5, 15, 5, 5, 5, 5, 5], dtype=float),
            live_time=10.0,
        )
        background = SpectrumRecord(
            filepath="background.spe",
            channels=np.arange(8, dtype=float),
            counts=np.full(8, 10.0),
            live_time=10.0,
        )
        net, info = subtract_background(sample, background)
        self.assertEqual(info.clipped_channels, 7)
        self.assertAlmostEqual(info.clipped_fraction, 7 / 8)
        self.assertTrue(np.all(net.counts >= 0))
        self.assertAlmostEqual(float(net.counts[2]), 5.0)
        self.assertTrue(any("截断" in warning for warning in net.warnings))

    def test_different_channel_grid_is_resampled_with_warning(self) -> None:
        sample = SpectrumRecord(
            filepath="sample.spe",
            channels=np.arange(10, dtype=float),
            counts=np.full(10, 20.0),
            live_time=10.0,
        )
        background = SpectrumRecord(
            filepath="background.spe",
            channels=np.arange(0, 20, 2, dtype=float),
            counts=np.full(10, 10.0),
            live_time=10.0,
        )
        net, info = subtract_background(sample, background)
        self.assertTrue(info.resampled)
        np.testing.assert_allclose(net.counts, np.full(10, 10.0))
        self.assertTrue(any("重采样" in warning for warning in net.warnings))

    def test_non_overlapping_ranges_are_rejected(self) -> None:
        sample = SpectrumRecord(
            filepath="sample.spe",
            channels=np.arange(10, dtype=float),
            counts=np.full(10, 1.0),
        )
        background = SpectrumRecord(
            filepath="background.spe",
            channels=np.arange(100, 110, dtype=float),
            counts=np.full(10, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "没有重叠"):
            subtract_background(sample, background)

    def test_measured_background_removes_an_environmental_line(self) -> None:
        rng = np.random.default_rng(7)
        channels = np.arange(4096, dtype=float)
        continuum = 4.0 + 16.0 * np.exp(-channels / 850.0)

        def build(line_amplitudes: tuple[tuple[float, float, float], ...]) -> np.ndarray:
            expected = continuum.copy()
            for energy, amplitude, sigma in line_amplitudes:
                center = energy / DEMO_SLOPE_KEV_PER_CHANNEL
                expected += amplitude * np.exp(-0.5 * ((channels - center) / sigma) ** 2)
            return rng.poisson(expected).astype(float)

        sample = SpectrumRecord(
            filepath="sample.spe",
            channels=channels,
            counts=build(((1173.237, 900.0, 5.5), (1332.501, 820.0, 5.8), (1460.822, 700.0, 6.0))),
            live_time=295.0,
            real_time=300.0,
        )
        background = SpectrumRecord(
            filepath="background.spe",
            channels=channels,
            counts=build(((1460.822, 700.0, 6.0),)),
            live_time=295.0,
            real_time=300.0,
        )
        calibration = CalibrationModel([DEMO_SLOPE_KEV_PER_CHANNEL, DEMO_OFFSET_KEV])

        def k40_area(peaks: list[PeakResult]) -> float:
            return sum(peak.area for peak in peaks if abs(peak.energy - 1460.822) < 10.0)

        raw = analyze_spectrum(sample, calibration)
        corrected = analyze_spectrum(sample, calibration, background=background)
        self.assertIsNotNone(corrected.background)
        self.assertAlmostEqual(corrected.background.scale, 1.0, places=6)
        self.assertGreater(k40_area(raw.peaks), 2000.0)
        self.assertLess(k40_area(corrected.peaks), k40_area(raw.peaks) * 0.3)
        co60 = [peak.primary_isotope for peak in corrected.peaks if peak.primary_isotope == "co-60"]
        self.assertGreaterEqual(len(co60), 2)

    def test_analyze_spectrum_exposes_background_metadata(self) -> None:
        record = synthetic_co60_record()
        calibration = CalibrationModel([DEMO_SLOPE_KEV_PER_CHANNEL, DEMO_OFFSET_KEV])
        background = SpectrumRecord(
            filepath="background.spe",
            channels=record.channels,
            counts=np.zeros_like(record.counts),
            live_time=record.live_time,
        )
        result = analyze_spectrum(record, calibration, background=background)
        self.assertIsNotNone(result.background)
        self.assertAlmostEqual(result.background.scale, 1.0, places=6)
        self.assertEqual(result.background.continuum, "snip")
        np.testing.assert_allclose(result.analysis_counts, record.counts)
        config = AnalysisConfig(snip_after_background=False)
        flat_result = analyze_spectrum(record, calibration, config=config, background=background)
        self.assertEqual(flat_result.background.continuum, "none")
        np.testing.assert_allclose(flat_result.baseline, np.zeros_like(record.counts))


class PeakFitTests(unittest.TestCase):
    def _gaussian_spectrum(self, lines, sigma, baseline=10.0, seed=3, length=220):
        rng = np.random.default_rng(seed)
        x = np.arange(length, dtype=float)
        expected = np.full(length, baseline, dtype=float)
        for centre, amplitude in lines:
            expected += amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
        return x, rng.poisson(expected).astype(float)

    def test_gaussian_fit_recovers_area_with_uncertainty(self) -> None:
        x, counts = self._gaussian_spectrum([(110.0, 400.0)], sigma=5.0)
        window = slice(80, 140)
        fit = fit_peak_shape(x[window], counts[window], 110.0, 5.0, 400.0)
        self.assertIsNotNone(fit)
        expected_area = 400.0 * 5.0 * np.sqrt(2.0 * np.pi)
        self.assertAlmostEqual(fit["area"], expected_area, delta=expected_area * 0.05)
        self.assertAlmostEqual(fit["center"], 110.0, delta=0.2)
        self.assertGreater(fit["area_uncertainty"], 0.0)
        self.assertLess(fit["area_uncertainty"], expected_area * 0.05)

    def test_close_peaks_are_merged(self) -> None:
        peaks = [
            PeakResult(channel=100.0, fwhm=10.0, area=300.0, snr=20.0, area_uncertainty=30.0),
            PeakResult(channel=102.0, fwhm=10.0, area=100.0, snr=15.0, area_uncertainty=20.0),
            PeakResult(channel=200.0, fwhm=10.0, area=50.0, snr=8.0),
        ]
        merged = merge_close_peaks(peaks)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].area, 400.0)
        self.assertAlmostEqual(merged[0].channel, 100.5)
        self.assertAlmostEqual(merged[0].area_uncertainty, math.hypot(30.0, 20.0))

    def test_doublet_decomposition_recovers_separated_components(self) -> None:
        x, counts = self._gaussian_spectrum([(90.0, 300.0), (99.0, 200.0)], sigma=3.0)
        peak = PeakResult(channel=94.5, fwhm=3.0 * 2.35482, area=1000.0, snr=30.0)
        peak.fwhm_kev = peak.fwhm
        peak.interference = [
            InterferenceLine(isotope="alpha", energy=90.0, intensity=0.5, delta_kev=-4.5, weight=0.5),
            InterferenceLine(isotope="beta", energy=99.0, intensity=0.5, delta_kev=4.5, weight=0.5),
        ]
        decompose_interference([peak], x, counts, CalibrationModel([1.0, 0.0]))
        self.assertTrue(peak.decomposition_resolved)
        self.assertEqual([component.isotope for component in peak.components], ["alpha", "beta"])
        areas = {component.isotope: component.area for component in peak.components}
        expected = {isotope: amplitude * 3.0 * np.sqrt(2.0 * np.pi) for isotope, amplitude in (("alpha", 300.0), ("beta", 200.0))}
        for isotope, value in expected.items():
            self.assertAlmostEqual(areas[isotope], value, delta=value * 0.25)
        self.assertAlmostEqual(sum(component.share for component in peak.components), 1.0, places=6)

    def test_doublet_decomposition_is_rejected_when_blended(self) -> None:
        x, counts = self._gaussian_spectrum([(95.0, 400.0), (96.0, 380.0)], sigma=4.0)
        peak = PeakResult(channel=95.5, fwhm=4.0 * 2.35482, area=4000.0, snr=40.0)
        peak.fwhm_kev = peak.fwhm
        peak.interference = [
            InterferenceLine(isotope="alpha", energy=95.0, intensity=0.5, delta_kev=-0.5, weight=0.5),
            InterferenceLine(isotope="beta", energy=96.0, intensity=0.5, delta_kev=0.5, weight=0.5),
        ]
        decompose_interference([peak], x, counts, CalibrationModel([1.0, 0.0]))
        self.assertFalse(peak.decomposition_resolved)
        self.assertIn("FWHM", peak.decomposition_note)
        self.assertEqual(len(peak.components), 2)


if __name__ == "__main__":
    unittest.main()
