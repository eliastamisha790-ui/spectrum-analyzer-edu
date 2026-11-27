"""Educational gamma spectrum analysis toolkit."""

from .core import (
    AnalysisConfig,
    BackgroundInfo,
    CalibrationModel,
    EfficiencyModel,
    PeakResult,
    SpectrumRecord,
    analyze_spectrum,
    fit_energy_calibration,
    load_nuclide_library,
    load_spectrum_file,
    subtract_background,
)

__all__ = [
    "AnalysisConfig",
    "BackgroundInfo",
    "CalibrationModel",
    "EfficiencyModel",
    "PeakResult",
    "SpectrumRecord",
    "analyze_spectrum",
    "fit_energy_calibration",
    "load_nuclide_library",
    "load_spectrum_file",
    "subtract_background",
]
__version__ = "0.4.0"
