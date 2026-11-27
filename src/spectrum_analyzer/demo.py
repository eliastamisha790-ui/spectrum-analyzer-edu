"""Deterministic synthetic spectra for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import SpectrumRecord


DEMO_SLOPE_KEV_PER_CHANNEL = 0.4
DEMO_OFFSET_KEV = 0.0

# 16384 道、约 2245 keV 量程，峰宽 1.4 keV，与常见教学 HPGe 谱接近
DOUBLET_SLOPE_KEV_PER_CHANNEL = 0.137145
DOUBLET_OFFSET_KEV = 0.0133
DOUBLET_FWHM_KEV = 1.4


def synthetic_doublet_record(seed: int = 2028) -> SpectrumRecord:
    """Synthetic U-238-series spectrum whose 186 keV peak is an unresolved doublet.

    U-235 (185.715 keV, 57.2% per decay) and Ra-226 (186.211 keV, 3.64%) are only
    0.5 keV apart, while the modelled detector FWHM is 1.4 keV — the exact
    situation the doublet logic exists for. Line strengths follow an equal-activity
    natural-uranium assumption (A(U-235)/A(U-238) = 4.6%), so the U-235 line
    carries 0.0264 photon/decay against 0.0364 for Ra-226.
    """

    rng = np.random.default_rng(seed)
    channels = np.arange(16384, dtype=float)
    continuum = 6.0 + 260.0 * np.exp(-channels / 700.0)
    sigma_channels = (DOUBLET_FWHM_KEV / 2.35482) / DOUBLET_SLOPE_KEV_PER_CHANNEL
    expected = continuum.copy()
    for energy, emission_per_decay in (
        (185.715, 0.572 * 0.046),
        (186.211, 0.0364),
        (241.997, 0.0727),
        (295.224, 0.193),
        (351.932, 0.376),
        (609.312, 0.457),
    ):
        center = (energy - DOUBLET_OFFSET_KEV) / DOUBLET_SLOPE_KEV_PER_CHANNEL
        amplitude = 60000.0 * emission_per_decay / (sigma_channels * np.sqrt(2.0 * np.pi))
        expected += amplitude * np.exp(-0.5 * ((channels - center) / sigma_channels) ** 2)
    counts = rng.poisson(np.clip(expected, 0, None)).astype(float)
    return SpectrumRecord(
        filepath="synthetic-doublet.spe",
        channels=channels,
        counts=counts,
        live_time=1800.0,
        real_time=1815.0,
    )


def synthetic_co60_record(seed: int = 2025) -> SpectrumRecord:
    """Create a clearly labeled synthetic Co-60-like teaching spectrum."""

    rng = np.random.default_rng(seed)
    channels = np.arange(4096, dtype=float)
    background = 4.0 + 16.0 * np.exp(-channels / 850.0)
    expected = background.copy()
    for energy, amplitude, sigma in (
        (1173.237, 900.0, 5.5),
        (1332.501, 820.0, 5.8),
    ):
        center = (energy - DEMO_OFFSET_KEV) / DEMO_SLOPE_KEV_PER_CHANNEL
        expected += amplitude * np.exp(-0.5 * ((channels - center) / sigma) ** 2)
    counts = rng.poisson(np.clip(expected, 0, None)).astype(float)
    return SpectrumRecord(
        filepath="synthetic-co60-demo.spe",
        channels=channels,
        counts=counts,
        live_time=295.0,
        real_time=300.0,
    )


def synthetic_sample_with_k40_record(seed: int = 2026) -> SpectrumRecord:
    """Co-60 lines plus an environmental K-40 line, for background-subtraction demos."""

    rng = np.random.default_rng(seed)
    channels = np.arange(4096, dtype=float)
    background = 4.0 + 16.0 * np.exp(-channels / 850.0)
    expected = background.copy()
    for energy, amplitude, sigma in (
        (1173.237, 900.0, 5.5),
        (1332.501, 820.0, 5.8),
        (1460.822, 700.0, 6.0),
    ):
        center = (energy - DEMO_OFFSET_KEV) / DEMO_SLOPE_KEV_PER_CHANNEL
        expected += amplitude * np.exp(-0.5 * ((channels - center) / sigma) ** 2)
    counts = rng.poisson(np.clip(expected, 0, None)).astype(float)
    return SpectrumRecord(
        filepath="synthetic-sample-with-k40.spe",
        channels=channels,
        counts=counts,
        live_time=295.0,
        real_time=300.0,
    )


def synthetic_lab_background_record(seed: int = 2027) -> SpectrumRecord:
    """Half-length background measurement with the same continuum and K-40 line.

    Counts are half of the sample's because the live time is half; dividing the
    sample live time by this live time gives the 1.97 scaling used on import.
    """

    rng = np.random.default_rng(seed)
    channels = np.arange(4096, dtype=float)
    expected = 4.0 + 16.0 * np.exp(-channels / 850.0)
    center = (1460.822 - DEMO_OFFSET_KEV) / DEMO_SLOPE_KEV_PER_CHANNEL
    expected += 700.0 * np.exp(-0.5 * ((channels - center) / 6.0) ** 2)
    counts = rng.poisson(np.clip(0.5 * expected, 0, None)).astype(float)
    return SpectrumRecord(
        filepath="synthetic-lab-background.spe",
        channels=channels,
        counts=counts,
        live_time=150.0,
        real_time=151.0,
    )


def write_spe(record: SpectrumRecord, destination: str | Path) -> Path:
    """Write a minimal SPE file using live-time then real-time ordering."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = int(record.channels[0])
    end = int(record.channels[-1])
    live_time = record.live_time or 0.0
    real_time = record.real_time or live_time
    lines = [
        "$SPEC_ID:",
        "Synthetic Co-60 teaching spectrum (not measured data)",
        "$MEAS_TIM:",
        f"{live_time:.3f} {real_time:.3f}",
        "$DATA:",
        f"{start} {end}",
        *[str(int(value)) for value in record.counts],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    destination = Path(__file__).resolve().parents[2] / "examples"
    print(write_spe(synthetic_co60_record(), destination / "synthetic_co60.spe"))
    print(write_spe(synthetic_sample_with_k40_record(), destination / "synthetic_sample_with_k40.spe"))
    print(write_spe(synthetic_lab_background_record(), destination / "synthetic_lab_background.spe"))
    print(write_spe(synthetic_doublet_record(), destination / "synthetic_doublet.spe"))


if __name__ == "__main__":
    main()
