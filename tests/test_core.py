from __future__ import annotations

import math
import tempfile
import unittest
import zipfile
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
    auto_calibrate,
    decompose_interference,
    detect_peaks_from_cwt,
    fit_energy_calibration,
    fit_peak_shape,
    flatten_library,
    load_nuclide_library,
    load_spectrum_file,
    match_nuclides,
    merge_close_peaks,
    refine_peaks,
    snip_baseline,
    subtract_background,
    summarize_nuclide_matches,
)
from spectrum_analyzer.certificate import (
    certificate_from_dict,
    certificate_library_from_file,
    certificate_library_from_text,
    normalize_nuclide,
    parse_certificate_text,
    read_certificate_file,
)
from spectrum_analyzer.efficiency import (
    build_efficiency_from_certificate,
    build_line_efficiencies,
    quantify_sample,
    summarize_activities,
)
from spectrum_analyzer.demo import (
    DEMO_OFFSET_KEV,
    DEMO_SLOPE_KEV_PER_CHANNEL,
    DOUBLET_OFFSET_KEV,
    DOUBLET_SLOPE_KEV_PER_CHANNEL,
    synthetic_co60_record,
    synthetic_doublet_record,
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

    def test_synthetic_doublet_example_is_flagged_end_to_end(self) -> None:
        record = synthetic_doublet_record()
        result = analyze_spectrum(
            record,
            CalibrationModel([DOUBLET_SLOPE_KEV_PER_CHANNEL, DOUBLET_OFFSET_KEV]),
        )
        doublets = [peak for peak in result.peaks if peak.is_ambiguous]
        self.assertEqual(len(doublets), 1)
        self.assertAlmostEqual(doublets[0].energy, 186.0, delta=0.6)
        self.assertEqual(
            {line.isotope for line in doublets[0].interference},
            {"u-235", "ra-226"},
        )
        self.assertEqual(doublets[0].primary_isotope, "")
        self.assertGreater(doublets[0].area_uncertainty, 0.0)


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


class AutoCalibrationTests(unittest.TestCase):
    LIBRARY = {name: NNDC_LIBRARY[name] for name in ("eu-152", "co-60", "cs-137")}
    MIXED_LINES = (
        (121.782, 3000.0),
        (244.697, 1200.0),
        (344.278, 4000.0),
        (778.903, 2000.0),
        (964.055, 2200.0),
        (1112.116, 2000.0),
        (1408.006, 2800.0),
        (1173.237, 15000.0),
        (1332.501, 14000.0),
        (661.657, 12000.0),
    )

    def _mixed_spectrum(self, slope: float, offset: float, sigma: float = 2.5, seed: int = 1) -> SpectrumRecord:
        rng = np.random.default_rng(seed)
        channels = np.arange(4096, dtype=float)
        expected = 3.0 + 40.0 * np.exp(-channels / 900.0)
        for energy, area in self.MIXED_LINES:
            centre = (energy - offset) / slope
            amplitude = area / (sigma * np.sqrt(2.0 * np.pi))
            expected += amplitude * np.exp(-0.5 * ((channels - centre) / sigma) ** 2)
        return SpectrumRecord(
            filepath="mixed.spe",
            channels=channels,
            counts=rng.poisson(np.clip(expected, 0, None)).astype(float),
            live_time=900.0,
            real_time=880.0,
        )

    @staticmethod
    def _peaks_of(record: SpectrumRecord) -> list[PeakResult]:
        net = np.clip(record.counts - snip_baseline(record.counts), 0, None)
        candidates = detect_peaks_from_cwt(net)
        return merge_close_peaks(refine_peaks(record.channels, net, record.counts, candidates))

    def test_recovers_mixed_source_calibration(self) -> None:
        record = self._mixed_spectrum(0.5, 20.0)
        peaks = self._peaks_of(record)
        self.assertGreaterEqual(len(peaks), 8)
        result = auto_calibrate(peaks, self.LIBRARY)
        self.assertIsNotNone(result.model)
        slope, offset = result.model.coefficients
        self.assertAlmostEqual(slope, 0.5, delta=0.001)
        self.assertAlmostEqual(offset, 20.0, delta=0.3)
        self.assertTrue(result.credible)
        self.assertFalse(result.ambiguous)
        identified = {name for name, _, _ in result.nuclide_ranking}
        self.assertEqual(identified, {"eu-152", "co-60", "cs-137"})

    def test_recovers_a_different_slope_range(self) -> None:
        record = self._mixed_spectrum(0.25, 0.0, sigma=1.5, seed=7)
        result = auto_calibrate(self._peaks_of(record), self.LIBRARY)
        self.assertIsNotNone(result.model)
        self.assertAlmostEqual(result.model.coefficients[0], 0.25, delta=0.001)
        self.assertAlmostEqual(result.model.coefficients[1], 0.0, delta=0.3)

    def test_requires_at_least_four_peaks(self) -> None:
        record = self._mixed_spectrum(0.5, 20.0)
        peaks = self._peaks_of(record)[:3]
        result = auto_calibrate(peaks, self.LIBRARY)
        self.assertIsNone(result.model)
        self.assertIn("4", result.note)

    def test_random_spectrum_is_not_credible(self) -> None:
        rng = np.random.default_rng(11)
        noise = SpectrumRecord(
            filepath="noise.spe",
            channels=np.arange(4096, dtype=float),
            counts=rng.poisson(5.0, 4096).astype(float),
            live_time=100.0,
            real_time=99.0,
        )
        result = auto_calibrate(self._peaks_of(noise), NNDC_LIBRARY)
        self.assertFalse(result.credible)

    def test_doublet_peak_contributes_only_one_fit_point(self) -> None:
        # 186 keV 处 U-235 与 Ra-226 共用同一条实测峰，若两条线都参与拟合会把刻度拉偏
        record = synthetic_doublet_record()
        peaks = self._peaks_of(record)
        result = auto_calibrate(peaks, NNDC_LIBRARY)
        self.assertIsNotNone(result.model)
        slope, offset = result.model.coefficients
        self.assertAlmostEqual(slope, DOUBLET_SLOPE_KEV_PER_CHANNEL, delta=0.0005)
        self.assertAlmostEqual(offset, DOUBLET_OFFSET_KEV, delta=0.15)


class CertificateTests(unittest.TestCase):
    TEXT = """标准源：GBW08304a（300g）
核素比活度 (Bq/kg)
238U：158
232Th：65.6
226Ra：132
40K：535
60Co：137
137Cs：81
定值时间：1995年1月31日
"""

    def test_parses_certificate_entries_units_and_metadata(self) -> None:
        certificate = parse_certificate_text(self.TEXT)
        self.assertEqual(len(certificate.entries), 6)
        by_nuclide = {entry.nuclide: entry for entry in certificate.entries}
        self.assertAlmostEqual(by_nuclide["u-238"].activity_bq_per_kg, 158.0)
        self.assertAlmostEqual(by_nuclide["th-232"].activity_bq_per_kg, 65.6)
        self.assertAlmostEqual(by_nuclide["cs-137"].activity_bq_per_kg, 81.0)
        self.assertEqual(certificate.reference_date, "1995-01-31")
        self.assertAlmostEqual(certificate.mass_kg, 0.3)

    def test_nuclide_label_normalisation(self) -> None:
        self.assertEqual(normalize_nuclide("238U"), "u-238")
        self.assertEqual(normalize_nuclide("U-238"), "u-238")
        self.assertEqual(normalize_nuclide("Cs-137"), "cs-137")
        self.assertEqual(normalize_nuclide("137Cs"), "cs-137")
        self.assertEqual(normalize_nuclide("K40"), "k-40")
        self.assertIsNone(normalize_nuclide("abc"))

    def test_daughters_replace_parents_and_drop_unrelated_nuclides(self) -> None:
        result = certificate_library_from_text(self.TEXT, NNDC_LIBRARY)
        self.assertEqual(
            set(result.nuclides),
            {
                "th-234",
                "u-235",
                "ac-228",
                "pb-212",
                "tl-208",
                "bi-212",
                "ra-226",
                "pb-214",
                "bi-214",
                "pb-210",
                "k-40",
                "co-60",
                "cs-137",
            },
        )
        # 证书里没有的核素必须被剔除，否则候选名单会被污染
        for absent in ("am-241", "cd-109", "co-57", "eu-152", "ba-133"):
            self.assertNotIn(absent, result.library)
        self.assertLess(result.line_count, sum(len(v) for v in NNDC_LIBRARY.values()))

    def test_every_kept_line_comes_from_the_base_library(self) -> None:
        result = certificate_library_from_text(self.TEXT, NNDC_LIBRARY)
        for isotope, lines in result.library.items():
            self.assertEqual(lines, NNDC_LIBRARY[isotope])
            for line in lines:
                self.assertGreater(line["energy"], 0)
                self.assertGreater(line["intensity"], 0)

    def test_unknown_nuclide_is_reported_instead_of_silently_dropped(self) -> None:
        result = certificate_library_from_text("238U：158\nTc-99：12\n", NNDC_LIBRARY)
        self.assertIn("tc-99", result.missing_proxies)
        self.assertIn("th-234", result.nuclides)

    def test_text_without_nuclides_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有从文本中识别出任何核素"):
            parse_certificate_text("这是一段没有任何核素的说明文字")

    def test_activity_is_scaled_by_certificate_mass(self) -> None:
        certificate = parse_certificate_text(self.TEXT)
        entry = next(item for item in certificate.entries if item.nuclide == "cs-137")
        self.assertAlmostEqual(entry.activity_bq(certificate.mass_kg), 24.3, places=6)
        self.assertIsNone(entry.activity_bq(None))

    def test_reference_date_prefers_the_certification_line(self) -> None:
        text = (
            "标样测量日期2022年11月11日\n"
            "238U：158 Bq/kg\n"
            "检定时间1995年1月31日\n"
        )
        certificate = parse_certificate_text(text)
        self.assertEqual(certificate.reference_date, "1995-01-31")

    def test_decay_correction_matches_hand_calculation(self) -> None:
        text = "60Co：137 Bq/kg\n137Cs：81 Bq/kg\n检定时间：1995-01-31\n"
        result = certificate_library_from_text(text, NNDC_LIBRARY, target_date="2022-11-11")
        entries = {entry.nuclide: entry for entry in result.certificate.entries}
        self.assertAlmostEqual(entries["co-60"].decay_factor, 0.0259, places=3)
        self.assertAlmostEqual(entries["cs-137"].decay_factor, 0.527, places=3)
        self.assertAlmostEqual(entries["co-60"].corrected_bq_per_kg(), 3.55, places=1)
        self.assertAlmostEqual(entries["cs-137"].corrected_bq_per_kg(), 42.7, places=1)
        self.assertIn("衰减系数", result.summary_text())

    def test_docx_certificate_is_read_without_extra_dependencies(self) -> None:
        document = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            "<w:p><w:r><w:t>标准样品源总重300g</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>238U：158 Bq/kg</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>137Cs：81 Bq/kg</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>检定时间1995年1月31日</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cert.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document)
            result = certificate_library_from_file(str(path), NNDC_LIBRARY)
        self.assertEqual(len(result.certificate.entries), 2)
        self.assertEqual(result.certificate.reference_date, "1995-01-31")
        self.assertAlmostEqual(result.certificate.mass_kg, 0.3)
        self.assertIn("th-234", result.nuclides)

    def test_image_certificate_explains_the_ocr_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "label.jpg"
            path.write_bytes(b"\xff\xd8\xff\xe0JFIF")
            with self.assertRaisesRegex(ValueError, "OCR"):
                read_certificate_file(str(path))

    def test_spe_measurement_date_is_captured(self) -> None:
        content = (
            "$SPEC_ID:\ntest\n$DATE_MEA:\n11/11/2022 22:10:26\n"
            "$MEAS_TIM:\n100 110\n$DATA:\n0 3\n1\n2\n3\n4\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dated.spe"
            path.write_text(content, encoding="utf-8")
            record = load_spectrum_file(str(path))
        self.assertEqual(record.measurement_date, "2022-11-11")


class SpcFormatTests(unittest.TestCase):
    def _write_spc(self, path: Path, counts, header_text: str) -> None:
        header = header_text.encode("latin-1")
        header = header.ljust(7680, b"\x00")
        body = b"".join(int(value).to_bytes(4, "little") for value in counts)
        path.write_bytes(header + body)

    def test_spc_matches_the_spe_export(self) -> None:
        counts = [0, 1, 2, 3, 5, 8, 13, 21]
        header = "300G.Spc        11-Nov-221  22:10:26      139321    139453 CN-PL-DZ-01-33-34"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.spc"
            self._write_spc(path, counts, header)
            record = load_spectrum_file(str(path))
        np.testing.assert_array_equal(record.counts, counts)
        self.assertEqual(record.live_time, 139321.0)
        self.assertEqual(record.real_time, 139453.0)
        self.assertEqual(record.measurement_date, "2022-11-11")

    def test_spc_with_a_broken_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.spc"
            path.write_bytes(b"\x00" * 700)
            with self.assertRaisesRegex(ValueError, "SPC"):
                load_spectrum_file(str(path))


class EfficiencyTests(unittest.TestCase):
    LIVE_TIME = 1000.0

    def _certificate(self, activity_per_kg: float = 100.0):
        return certificate_from_dict(
            {
                "reference_date": "2000-01-01",
                "mass_kg": 1.0,
                "entries": [
                    {"label": "137Cs", "nuclide": "cs-137", "activity_bq_per_kg": activity_per_kg, "proxies": ["cs-137"]},
                    {"label": "40K", "nuclide": "k-40", "activity_bq_per_kg": activity_per_kg, "proxies": ["k-40"]},
                    {"label": "60Co", "nuclide": "co-60", "activity_bq_per_kg": activity_per_kg, "proxies": ["co-60"]},
                    {"label": "152Eu", "nuclide": "eu-152", "activity_bq_per_kg": activity_per_kg, "proxies": ["eu-152"]},
                ],
            }
        )

    def _peaks(self, efficiency_at, activity: float = 100.0, scale: dict | None = None):
        lines = [
            ("cs-137", 661.657, 0.851),
            ("co-60", 1173.237, 0.999),
            ("co-60", 1332.501, 0.999),
            ("k-40", 1460.822, 0.1066),
            ("eu-152", 244.697, 0.0755),
            ("eu-152", 344.278, 0.265),
            ("eu-152", 778.903, 0.1297),
            ("eu-152", 964.055, 0.145),
        ]
        peaks = []
        for isotope, energy, intensity in lines:
            efficiency = efficiency_at(energy)
            area = efficiency * activity * intensity * self.LIVE_TIME
            factor = (scale or {}).get(energy, 1.0)
            peak = PeakResult(channel=energy, energy=energy, fwhm=2.0, area=area * factor, snr=50.0)
            peak.primary_isotope = isotope
            peak.reference_intensity = intensity
            peak.area_uncertainty = area * 0.01
            peaks.append(peak)
        return peaks

    def test_recovers_a_known_efficiency_curve(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        result = build_efficiency_from_certificate(
            self._peaks(truth),
            self._certificate(),
            self.LIVE_TIME,
            NNDC_LIBRARY,
        )
        self.assertIsNotNone(result.model)
        self.assertEqual(len([point for point in result.points if point.accepted]), 8)
        for energy in (600.0, 1000.0, 1400.0):
            self.assertAlmostEqual(result.model.efficiency(energy), truth(energy), delta=truth(energy) * 0.03)
        self.assertLess(result.relative_rms, 0.02)

    def test_reported_residuals_match_the_returned_model(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        result = build_efficiency_from_certificate(
            self._peaks(truth, scale={778.903: 1.4}),
            self._certificate(),
            self.LIVE_TIME,
            NNDC_LIBRARY,
        )
        self.assertIsNotNone(result.model)
        for point in result.points:
            predicted = result.model.efficiency(point.energy)
            self.assertAlmostEqual(point.residual, math.log(point.efficiency / predicted), places=6)
        self.assertTrue(any(not point.accepted for point in result.points) or result.relative_rms < 0.15)

    def test_missing_live_time_is_reported(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        result = build_efficiency_from_certificate(self._peaks(truth), self._certificate(), None, NNDC_LIBRARY)
        self.assertIsNone(result.model)
        self.assertIn("活时间", result.note)

    def test_ambiguous_peaks_are_skipped(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        peaks = self._peaks(truth)
        peaks[0].interference = [
            InterferenceLine(isotope="cs-137", energy=661.657, intensity=0.851, delta_kev=0.0, weight=1.0),
            InterferenceLine(isotope="other", energy=661.9, intensity=0.5, delta_kev=0.2, weight=0.5),
        ]
        result = build_efficiency_from_certificate(peaks, self._certificate(), self.LIVE_TIME, NNDC_LIBRARY)
        self.assertTrue(any("双线" in item for item in result.skipped))
        self.assertEqual(len(result.points), 7)


    def test_per_line_efficiency_quantifies_the_same_nuclide(self) -> None:
        """标准源样品同几何时，用同一核素自己的线定效率应能还原输入活度。"""

        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        standard_peaks = self._peaks(truth)
        lines = build_line_efficiencies(standard_peaks, self._certificate(), self.LIVE_TIME)
        self.assertEqual(len(lines), len(standard_peaks))

        sample_mass = 0.5
        sample_peaks = []
        for peak in standard_peaks:
            clone = PeakResult(
                channel=peak.channel,
                energy=peak.energy,
                fwhm=2.0,
                area=peak.area * 0.1,
                snr=20.0,
            )
            clone.primary_isotope = peak.primary_isotope
            clone.reference_intensity = peak.reference_intensity
            clone.area_uncertainty = clone.area * 0.02
            sample_peaks.append(clone)

        rows = quantify_sample(sample_peaks, lines, self.LIVE_TIME, sample_mass).rows
        summary = {item["isotope"]: item for item in summarize_activities(rows)}
        # 样品活度是标准的 1/10，质量 0.5 kg → Bq/kg = 100 × 标准 Bq/kg(100)
        for isotope in ("cs-137", "co-60", "k-40"):
            self.assertAlmostEqual(summary[isotope]["activity_bq_per_kg"], 20.0, delta=0.5)

    def test_per_line_efficiency_keeps_cascade_lines_apart(self) -> None:
        """级联线被符合相加压低时，逐线法不会把偏差摊到别的核素头上。"""

        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        standard = self._peaks(truth, scale={1332.501: 0.5})
        lines = build_line_efficiencies(standard, self._certificate(), self.LIVE_TIME)
        by_energy = {round(item.energy, 1): item.efficiency for item in lines}
        self.assertLess(by_energy[1332.5], by_energy[1173.2] * 0.6)


    def test_low_energy_lines_are_flagged_not_silently_reported(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        certificate = certificate_from_dict(
            {
                "reference_date": None,
                "mass_kg": 1.0,
                "entries": [
                    {"label": "137Cs", "nuclide": "cs-137", "activity_bq_per_kg": 100.0, "proxies": ["cs-137"]},
                    {"label": "210Pb", "nuclide": "pb-210", "activity_bq_per_kg": 100.0, "proxies": ["pb-210"]},
                ],
            }
        )
        standard = []
        for isotope, energy, intensity in (("cs-137", 661.657, 0.851), ("pb-210", 46.539, 0.0425)):
            area = truth(energy) * 100.0 * intensity * self.LIVE_TIME
            peak = PeakResult(channel=energy, energy=energy, fwhm=2.0, area=area, snr=50.0)
            peak.primary_isotope = isotope
            peak.reference_intensity = intensity
            peak.area_uncertainty = area * 0.01
            standard.append(peak)
        lines = build_line_efficiencies(standard, certificate, self.LIVE_TIME)

        low = PeakResult(channel=46.6, energy=46.6, fwhm=2.0, area=1000.0, snr=30.0)
        low.primary_isotope = "pb-210"
        low.reference_intensity = 0.0425
        low.area_uncertainty = 10.0
        far = PeakResult(channel=700.0, energy=700.0, fwhm=2.0, area=1000.0, snr=30.0)
        far.primary_isotope = "cs-137"
        far.reference_intensity = 0.851
        far.area_uncertainty = 10.0

        quantified = quantify_sample([low, far], lines, self.LIVE_TIME, 0.3)
        self.assertEqual(len(quantified.rows), 2)
        # 两条都保留数值，但必须带外推标记，不能默默给值
        self.assertTrue(all(row.flag for row in quantified.rows))
        self.assertEqual(len(quantified.flagged), 2)
        self.assertTrue(any("外推" in item for item in quantified.flagged))

    def test_uncertainty_combines_certificate_and_area(self) -> None:
        def truth(energy: float) -> float:
            return 0.02 * (energy / 1000.0) ** -0.5

        lines = build_line_efficiencies(self._peaks(truth), self._certificate(), self.LIVE_TIME)
        quantified = quantify_sample(self._peaks(truth), lines, self.LIVE_TIME, 1.0)
        for row in quantified.rows:
            # 面积统计 1% + 证书默认 5% → 合成约 5.1%，不应停留在 1%
            self.assertGreater(row.relative_uncertainty, 0.045)
            self.assertLess(row.relative_uncertainty, 0.10)


if __name__ == "__main__":
    unittest.main()
