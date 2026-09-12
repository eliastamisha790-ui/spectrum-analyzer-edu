"""Deterministic synthetic spectra for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import SpectrumRecord


DEMO_SLOPE_KEV_PER_CHANNEL = 0.4
DEMO_OFFSET_KEV = 0.0


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
    destination = Path(__file__).resolve().parents[2] / "examples" / "synthetic_co60.spe"
    print(write_spe(synthetic_co60_record(), destination))


if __name__ == "__main__":
    main()
