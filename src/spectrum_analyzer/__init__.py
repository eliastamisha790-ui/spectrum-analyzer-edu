"""Educational gamma spectrum analysis toolkit."""

from .core import (
    AnalysisConfig,
    AutoCalibrationResult,
    BackgroundInfo,
    CalibrationModel,
    EfficiencyModel,
    PeakResult,
    SpectrumRecord,
    analyze_spectrum,
    auto_calibrate,
    fit_energy_calibration,
    load_nuclide_library,
    load_spectrum_file,
    subtract_background,
)

__all__ = [
    "AnalysisConfig",
    "AutoCalibrationResult",
    "BackgroundInfo",
    "CalibrationModel",
    "EfficiencyModel",
    "PeakResult",
    "SpectrumRecord",
    "analyze_spectrum",
    "auto_calibrate",
    "fit_energy_calibration",
    "load_nuclide_library",
    "load_spectrum_file",
    "subtract_background",
]
__version__ = "0.7.0"
