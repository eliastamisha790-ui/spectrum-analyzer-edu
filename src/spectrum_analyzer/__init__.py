"""Educational gamma spectrum analysis toolkit."""

from .core import (
    AnalysisConfig,
    CalibrationModel,
    EfficiencyModel,
    PeakResult,
    SpectrumRecord,
    analyze_spectrum,
    fit_energy_calibration,
    load_nuclide_library,
    load_spectrum_file,
)

__all__ = [
    "AnalysisConfig",
    "CalibrationModel",
    "EfficiencyModel",
    "PeakResult",
    "SpectrumRecord",
    "analyze_spectrum",
    "fit_energy_calibration",
    "load_nuclide_library",
    "load_spectrum_file",
]
__version__ = "0.3.0"
