"""Core spectrum I/O and analysis primitives.

The algorithms are intended for teaching and exploratory analysis. They are
not validated for safety decisions, regulatory reporting, or certified
radionuclide quantification.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:  # pragma: no cover - NumPy < 2 compatibility
    _trapz = np.trapz

XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XLSX_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SPECTRUM_FILE_FILTER = (
    "支持的谱文件 (*.spe *.Spe *.SPE *.xlsx *.XLSX *.csv *.CSV *.txt *.TXT);;"
    "Canberra SPE (*.spe *.Spe *.SPE);;"
    "Excel 工作簿 (*.xlsx *.XLSX);;"
    "文本数据 (*.csv *.CSV *.txt *.TXT);;"
    "所有文件 (*.*)"
)
CHANNEL_HEADER_KEYWORDS = (
    "channel",
    "channels",
    "chan",
    "index",
    "道址",
    "通道",
    "道号",
    "序号",
    "编号",
)
COUNT_HEADER_KEYWORDS = (
    "count",
    "counts",
    "计数",
    "强度",
    "intensity",
    "value",
    "y",
)
REAL_TIME_HEADER_KEYWORDS = (
    "realtime",
    "real",
    "实时间",
    "实时",
)
LIVE_TIME_HEADER_KEYWORDS = (
    "livetime",
    "live",
    "活时间",
    "死时间",
)

NNDC_LIBRARY: Dict[str, List[Dict[str, float]]] = {
  "co-60": [
    {"energy": 1173.237, "intensity": 0.999},
    {"energy": 1332.501, "intensity": 0.999}
  ],
  "cs-137": [
    {"energy": 661.657, "intensity": 0.851}
  ],
  "co-57":[{"energy":14.413,"intensity":0.0916},
           {"energy":122.061,"intensity":0.8560},
           {"energy":136.474,"intensity":0.1068}
           ],
  "ba-133": [
    {"energy": 80.997, "intensity": 0.342},
    {"energy": 276.398, "intensity": 0.0716},
    {"energy": 302.853, "intensity": 0.1834},
    {"energy": 356.017, "intensity": 0.622},
    {"energy": 383.851, "intensity": 0.0894}
  ],
  "na-22": [
    {"energy": 1274.537, "intensity": 0.999}
  ],
  "am-241": [
    {"energy": 59.541, "intensity": 0.359}
  ],
  "eu-152": [
    {"energy": 121.782, "intensity": 0.284},
    {"energy": 244.697, "intensity": 0.0755},
    {"energy": 344.278, "intensity": 0.265},
    {"energy": 778.903, "intensity": 0.1297},
    {"energy": 964.055, "intensity": 0.145},
    {"energy": 1112.116, "intensity": 0.1341},
    {"energy": 1408.006, "intensity": 0.2087}
  ],
  "k-40": [
    {"energy": 1460.822, "intensity": 0.1066}
  ],
  "mn-54": [
    {"energy": 834.848, "intensity": 0.9998}
  ],
  "y-88": [
    {"energy": 898.042, "intensity": 0.937},
    {"energy": 1836.063, "intensity": 0.9936}
  ],
  "bi-207": [
    {"energy": 569.698, "intensity": 0.977},
    {"energy": 1063.656, "intensity": 0.745}
  ],
  "hg-203": [
    {"energy": 279.195, "intensity": 0.815}
  ],
  "cd-109": [
    {"energy": 88.034, "intensity": 0.036}
  ],
  "i-131": [
    {"energy": 364.489, "intensity": 0.817}
  ],
  "ce-139": [
    {"energy": 165.857, "intensity": 0.799}
  ],
  "ru-106": [
    {"energy": 621.900, "intensity": 0.098}
  ],
  "sr-85": [
    {"energy": 514.007, "intensity": 0.984}
  ],
  "zn-65": [
    {"energy": 1115.546, "intensity": 0.506}
  ],
  # --- 天然放射性核素（U-238 系 / Th-232 系 / K-40）---
  # 参考线与发射概率取自 ENSDF（经 NuDat 3 复核），仅收录教学与筛查常用的
  # 主线；强度为每次衰变的绝对发射概率（γ/衰变）。
  # 注意：u-235 的 185.715 keV 与 ra-226 的 186.211 keV 在常规 HPGe/NaI 分辨率
  # 下不可分辨，二者必须由 match_nuclides 的双线逻辑标注，不能单独认领。
  "u-235": [
    {"energy": 143.76, "intensity": 0.1096},
    {"energy": 163.33, "intensity": 0.0508},
    {"energy": 185.715, "intensity": 0.572},
    {"energy": 205.31, "intensity": 0.0501}
  ],
  "ra-226": [
    {"energy": 186.211, "intensity": 0.0364}
  ],
  "pb-210": [
    {"energy": 46.539, "intensity": 0.0425}
  ],
  "th-234": [
    {"energy": 63.29, "intensity": 0.048},
    {"energy": 92.38, "intensity": 0.0281}
  ],
  "pb-214": [
    {"energy": 241.997, "intensity": 0.0727},
    {"energy": 295.224, "intensity": 0.193},
    {"energy": 351.932, "intensity": 0.376}
  ],
  "bi-214": [
    {"energy": 609.312, "intensity": 0.457},
    {"energy": 665.453, "intensity": 0.0151},
    {"energy": 768.357, "intensity": 0.0489},
    {"energy": 806.174, "intensity": 0.0121},
    {"energy": 934.061, "intensity": 0.0316},
    {"energy": 1120.294, "intensity": 0.151},
    {"energy": 1155.19, "intensity": 0.0163},
    {"energy": 1238.111, "intensity": 0.0594},
    {"energy": 1280.96, "intensity": 0.0143},
    {"energy": 1377.669, "intensity": 0.04},
    {"energy": 1407.98, "intensity": 0.0215},
    {"energy": 1661.28, "intensity": 0.0115},
    {"energy": 1729.6, "intensity": 0.029},
    {"energy": 1764.494, "intensity": 0.154},
    {"energy": 2118.55, "intensity": 0.0116},
    {"energy": 2204.21, "intensity": 0.049}
  ],
  "ac-228": [
    {"energy": 209.253, "intensity": 0.039},
    {"energy": 338.32, "intensity": 0.114},
    {"energy": 794.947, "intensity": 0.043},
    {"energy": 911.204, "intensity": 0.262},
    {"energy": 964.766, "intensity": 0.0499},
    {"energy": 968.971, "intensity": 0.162}
  ],
  "pb-212": [
    {"energy": 238.632, "intensity": 0.436},
    {"energy": 300.087, "intensity": 0.033}
  ],
  "bi-212": [
    {"energy": 727.33, "intensity": 0.0658},
    {"energy": 1620.5, "intensity": 0.0149}
  ],
  "tl-208": [
    {"energy": 510.77, "intensity": 0.226},
    {"energy": 583.187, "intensity": 0.85},
    {"energy": 860.53, "intensity": 0.125},
    {"energy": 2614.511, "intensity": 0.9975}
  ]
}


def validate_nuclide_library(payload: object) -> Dict[str, List[Dict[str, float]]]:
    """Validate and normalize a nuclide reference library mapping."""

    if not isinstance(payload, dict) or not payload:
        raise ValueError("核素库必须是非空对象，键为核素名称，值为参考线列表。")
    normalized: Dict[str, List[Dict[str, float]]] = {}
    for raw_name, raw_lines in payload.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("核素名称不能为空。")
        if not isinstance(raw_lines, list) or not raw_lines:
            raise ValueError(f"核素 {name} 必须至少包含一条参考线。")
        lines: List[Dict[str, float]] = []
        for index, raw_line in enumerate(raw_lines, 1):
            if not isinstance(raw_line, dict):
                raise ValueError(f"核素 {name} 的第 {index} 条参考线格式错误。")
            energy_value = raw_line.get("energy", raw_line.get("energy_kev"))
            intensity_value = raw_line.get("intensity", raw_line.get("emission_probability"))
            try:
                energy = float(energy_value)
                intensity = float(intensity_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"核素 {name} 的第 {index} 条参考线缺少有效数值。") from exc
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError(f"核素 {name} 的参考能量必须为正有限值。")
            if not math.isfinite(intensity) or intensity <= 0:
                raise ValueError(f"核素 {name} 的发射概率必须为正有限值。")
            lines.append({"energy": energy, "intensity": intensity})
        normalized[name] = sorted(lines, key=lambda item: item["energy"])
    return normalized


def load_nuclide_library(filepath: str) -> Dict[str, List[Dict[str, float]]]:
    """Load a reference library from JSON or a three-column CSV file."""

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取核素库 JSON：{exc}") from exc
        return validate_nuclide_library(payload)
    if suffix != ".csv":
        raise ValueError("核素库仅支持 JSON 或 CSV 文件。")

    text: Optional[str] = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ValueError(f"无法读取核素库 CSV：{exc}") from exc
    if text is None:
        raise ValueError("核素库 CSV 编码无法识别，请使用 UTF-8 或 GB18030。")

    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        raise ValueError("核素库 CSV 缺少表头。")
    headers = {name.strip().lower(): name for name in reader.fieldnames if name}

    def resolve_header(*aliases: str) -> Optional[str]:
        return next((headers[alias] for alias in aliases if alias in headers), None)

    isotope_header = resolve_header("nuclide", "isotope", "核素")
    energy_header = resolve_header("energy_kev", "energy", "能量", "能量(kev)")
    intensity_header = resolve_header("intensity", "emission_probability", "发射概率", "强度")
    if not isotope_header or not energy_header or not intensity_header:
        raise ValueError("核素库 CSV 需要 nuclide、energy_kev、intensity 三列。")
    payload: Dict[str, List[Dict[str, object]]] = {}
    for row_index, row in enumerate(reader, 2):
        name = str(row.get(isotope_header, "")).strip()
        if not name:
            raise ValueError(f"核素库 CSV 第 {row_index} 行缺少核素名称。")
        payload.setdefault(name, []).append(
            {
                "energy": row.get(energy_header),
                "intensity": row.get(intensity_header),
            }
        )
    return validate_nuclide_library(payload)


@dataclass
class SpectrumRecord:
    filepath: str
    channels: np.ndarray
    counts: np.ndarray
    real_time: Optional[float] = None
    live_time: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    measurement_date: Optional[str] = None


@dataclass
class InterferenceLine:
    """One reference line that cannot be separated from a neighbouring line."""

    isotope: str
    energy: float
    intensity: float
    delta_kev: float
    weight: float


@dataclass
class PeakComponent:
    """One component of a multi-line fit inside a single observed peak."""

    isotope: str
    energy: float
    area: float
    area_uncertainty: float
    share: float
    share_uncertainty: float

    def as_text(self) -> str:
        return (
            f"{self.isotope} {self.energy:.3f} keV: {self.area:.1f}±{self.area_uncertainty:.1f} "
            f"({self.share * 100:.0f}%±{self.share_uncertainty * 100:.0f}%)"
        )


@dataclass
class PeakResult:
    channel: float
    fwhm: float
    area: float
    snr: float
    energy: float = 0.0
    nuclide_candidates: List[str] = field(default_factory=list)
    confidence: float = 0.0
    primary_isotope: str = ""
    reference_intensity: float = 0.0
    activity_bq: Optional[float] = None
    interference: List[InterferenceLine] = field(default_factory=list)
    fwhm_kev: float = 0.0
    area_uncertainty: float = 0.0
    components: List[PeakComponent] = field(default_factory=list)
    decomposition_resolved: bool = False
    decomposition_note: str = ""

    @property
    def is_ambiguous(self) -> bool:
        """True when two or more reference lines share this peak within its FWHM."""

        return len({line.isotope for line in self.interference}) > 1

    def interference_text(self) -> str:
        if not self.is_ambiguous:
            return ""
        return " / ".join(
            f"{line.isotope}({line.energy:.1f}, {line.weight * 100:.0f}%)" for line in self.interference
        )

    def as_row(self) -> List[str]:
        activity = format_activity(self.activity_bq)
        if self.is_ambiguous:
            candidate_text = "† " + "/".join(line.isotope for line in self.interference[:3])
        else:
            candidate_text = ", ".join(self.nuclide_candidates) if self.nuclide_candidates else "—"
        confidence_text = f"{self.confidence * 100:.1f}%" + ("†" if self.is_ambiguous else "")
        return [
            f"{self.channel:.1f}",
            f"{self.energy:.2f}",
            f"{self.area:.1f}",
            f"{self.fwhm:.2f}",
            candidate_text,
            confidence_text,
            activity,
        ]


@dataclass
class EfficiencyPoint:
    energy: float
    efficiency: float
    peak_channel: float
    peak_area: float
    counts_per_second: float
    intensity: float
    nuclide: str


@dataclass
class ActivitySummary:
    isotope: str
    activity_bq: float
    peak_count: int


@dataclass
class NuclideMatchSummary:
    isotope: str
    matched_lines: int
    total_lines: int
    intensity_coverage: float
    match_score: float
    ambiguous_lines: int = 0


@dataclass
class BackgroundInfo:
    """Provenance for a measured-background subtraction."""

    filepath: str
    scale_mode: str
    scale: float
    resampled: bool
    clipped_channels: int
    clipped_fraction: float
    continuum: str = "snip"
    warnings: List[str] = field(default_factory=list)

    def describe_scale(self) -> str:
        labels = {"live": "活时间归一", "real": "实时间归一", "none": "不做时间归一"}
        return labels.get(self.scale_mode, self.scale_mode)


@dataclass
class AnalysisResult:
    baseline: np.ndarray
    net_counts: np.ndarray
    peaks: List[PeakResult]
    nuclide_summary: List[NuclideMatchSummary] = field(default_factory=list)
    background: Optional[BackgroundInfo] = None
    analysis_counts: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable parameters for detector- and spectrum-specific analysis."""

    snip_iterations: Optional[int] = None
    wavelet_widths: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    detection_threshold_sigma: float = 1.0
    min_peak_distance: int = 6
    min_snr: float = 1.25
    absolute_tolerance_kev: float = 0.5
    relative_tolerance: float = 0.002
    max_candidates: int = 3
    width_tolerance_factor: float = 0.5
    interference_width_factor: float = 1.0
    background_scale: str = "live"
    snip_after_background: bool = True

    def validated(self) -> "AnalysisConfig":
        if self.snip_iterations is not None and not 1 <= int(self.snip_iterations) <= 4096:
            raise ValueError("SNIP 迭代次数必须在 1 到 4096 之间。")
        if not self.wavelet_widths or any(int(value) < 1 or int(value) > 256 for value in self.wavelet_widths):
            raise ValueError("小波宽度必须是 1 到 256 之间的正整数。")
        if not math.isfinite(self.detection_threshold_sigma) or self.detection_threshold_sigma < 0:
            raise ValueError("寻峰阈值系数必须是非负有限值。")
        if int(self.min_peak_distance) < 1:
            raise ValueError("最小峰间距必须至少为 1 个通道。")
        if not math.isfinite(self.min_snr) or self.min_snr < 0:
            raise ValueError("最小信噪比必须是非负有限值。")
        if not math.isfinite(self.absolute_tolerance_kev) or self.absolute_tolerance_kev <= 0:
            raise ValueError("绝对匹配容差必须是正有限值。")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0:
            raise ValueError("相对匹配容差必须是非负有限值。")
        if not 1 <= int(self.max_candidates) <= 20:
            raise ValueError("候选核素数量必须在 1 到 20 之间。")
        if not math.isfinite(self.width_tolerance_factor) or not 0.0 <= self.width_tolerance_factor <= 5.0:
            raise ValueError("峰宽容差系数必须在 0 到 5 之间。")
        if not math.isfinite(self.interference_width_factor) or not 0.0 <= self.interference_width_factor <= 10.0:
            raise ValueError("双线判定系数必须在 0 到 10 之间。")
        if self.background_scale not in {"live", "real", "none"}:
            raise ValueError("本底归一方式必须是 live、real 或 none。")
        return self

    def as_dict(self) -> Dict[str, object]:
        return {
            "snip_iterations": self.snip_iterations,
            "wavelet_widths": list(self.wavelet_widths),
            "detection_threshold_sigma": self.detection_threshold_sigma,
            "min_peak_distance": self.min_peak_distance,
            "min_snr": self.min_snr,
            "absolute_tolerance_kev": self.absolute_tolerance_kev,
            "relative_tolerance": self.relative_tolerance,
            "max_candidates": self.max_candidates,
            "width_tolerance_factor": self.width_tolerance_factor,
            "interference_width_factor": self.interference_width_factor,
            "background_scale": self.background_scale,
            "snip_after_background": self.snip_after_background,
        }


def format_activity(activity: Optional[float]) -> str:
    if activity is None or activity <= 0:
        return "—"
    if activity >= 1e9:
        return f"{activity / 1e9:.3f} GBq"
    if activity >= 1e6:
        return f"{activity / 1e6:.3f} MBq"
    if activity >= 1e3:
        return f"{activity / 1e3:.3f} kBq"
    return f"{activity:.2f} Bq"


class CalibrationModel:
    def __init__(self, coefficients: Optional[List[float]] = None) -> None:
        self.coefficients = coefficients or [1.0, 0.0]

    def energy(self, channel: float) -> float:
        degree = len(self.coefficients) - 1
        value = 0.0
        for idx, coef in enumerate(self.coefficients):
            power = degree - idx
            value += coef * (channel ** power)
        return value

    def slope(self, channel: float, step: float = 0.5) -> float:
        """Local keV-per-channel derivative, used to convert FWHM to keV."""

        return (self.energy(channel + step) - self.energy(channel - step)) / (2.0 * step)

    def describe(self) -> str:
        terms = []
        degree = len(self.coefficients) - 1
        for idx, coef in enumerate(self.coefficients):
            power = degree - idx
            if power == 0:
                terms.append(f"{coef:.6f}")
            elif power == 1:
                terms.append(f"{coef:.6f}·x")
            else:
                terms.append(f"{coef:.6f}·x^{power}")
        return " + ".join(terms)

    def is_default(self) -> bool:
        return (
            len(self.coefficients) == 2
            and abs(self.coefficients[0] - 1.0) < 1e-9
            and abs(self.coefficients[1]) < 1e-9
        )


def fit_energy_calibration(
    channels: List[float] | np.ndarray,
    energies: List[float] | np.ndarray,
    degree: int = 1,
) -> Tuple[CalibrationModel, float]:
    """Fit E(channel) and return the model plus RMSE in keV."""

    x = np.asarray(channels, dtype=float)
    y = np.asarray(energies, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("刻度通道与能量必须是一维且长度一致。")
    if len(x) < 2:
        raise ValueError("能量刻度至少需要 2 个峰。")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("刻度点包含 NaN 或无穷值。")
    if len(np.unique(x)) != len(x):
        raise ValueError("刻度通道不能重复。")
    degree = int(degree)
    if degree < 1 or degree >= len(x):
        raise ValueError(f"{len(x)} 个刻度点仅支持 1 到 {len(x) - 1} 阶拟合。")
    coefficients = np.polyfit(x, y, degree)
    predicted = np.polyval(coefficients, x)
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
    return CalibrationModel([float(value) for value in coefficients]), rmse


@dataclass
class CalibrationMatch:
    """One reference line that landed on an observed peak."""

    isotope: str
    energy: float
    intensity: float
    channel: float
    residual_kev: float
    quality: float


@dataclass
class AutoCalibrationResult:
    """Outcome of the mixed-source automatic energy calibration search."""

    model: Optional[CalibrationModel]
    score: float
    runner_up_score: float
    ambiguous: bool
    matched: List[CalibrationMatch] = field(default_factory=list)
    nuclide_ranking: List[Tuple[str, int, float]] = field(default_factory=list)
    hypotheses: int = 0
    note: str = ""
    residual_rms_kev: float = 0.0
    normalized_rms: float = 0.0
    credible: bool = False

    def summary_text(self) -> str:
        if self.model is None:
            return self.note or "未能找到自洽的能量刻度候选。"

        spots = self.matched
        unique_channels = len({round(item.channel, 3) for item in spots})
        quality = (sum(item.quality for item in spots) / len(spots)) if spots else 0.0
        ranking = "、".join(f"{isotope}({lines} 线)" for isotope, lines, _ in self.nuclide_ranking[:4])
        return (
            f"刻度：{self.model.describe()}\n"
            f"命中的参考线：{len(spots)} 条 / 独立峰 {unique_channels} 个\n"
            f"平均能量吻合度：{quality * 100:.1f}%\n"
            f"残差 RMS：{self.residual_rms_kev:.3f} keV（归一化 {self.normalized_rms:.2f}）\n"
            f"主要贡献核素：{ranking or '无'}\n"
            f"评估过的候选刻度：{self.hypotheses} 个\n"
            + (f"次优候选得分 {self.runner_up_score:.3f}（最高 {self.score:.3f}）\n" if self.runner_up_score > 0 else "")
            + (f"可信度：{'通过' if self.credible else '未通过'}\n")
            + (f"注意：{self.note}" if self.note else "")
        )


def _nearest_channel_distance(predicted: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """Distance from each predicted channel to the closest observed peak channel."""

    position = np.searchsorted(channels, predicted)
    position = np.clip(position, 0, len(channels) - 1)
    right = channels[position]
    left_index = np.clip(position - 1, 0, len(channels) - 1)
    left = channels[left_index]
    return np.minimum(np.abs(predicted - right), np.abs(predicted - left))


def auto_calibrate(
    peaks: List[PeakResult],
    library: Dict[str, List[Dict[str, float]]],
    absolute_tolerance_kev: float = 0.5,
    relative_tolerance: float = 0.002,
    max_peaks: int = 36,
    min_intensity: float = 0.01,
    pair_intensity: float = 0.05,
    slope_bounds: Tuple[float, float] = (0.02, 5.0),
    min_points: int = 3,
) -> AutoCalibrationResult:
    """Find an energy calibration for a mixed-nuclide spectrum.

    The human workflow is reproduced literally: every pair of observed peaks is
    paired with every pair of reference lines to form a calibration hypothesis,
    then *all* remaining reference lines vote on that hypothesis. A wrong pairing
    maps the certificate lines to empty channels and dies on the global vote,
    while the right pairing lights up several lines of several nuclides at once.

    Scoring is intensity weighted and rewards complete isotopes rather than
    isolated hits, so a coincidence that happens to catch three lines of one
    nuclide still loses to a calibration that explains a whole decay scheme.
    The best handful of hypotheses are then refitted by least squares on their
    matched points, which removes the quantisation of the two-point estimate.
    """

    reference = validate_nuclide_library(library)
    usable = [peak for peak in peaks if peak.channel > 0 and peak.area > 0]
    if len(usable) < 4:
        return AutoCalibrationResult(
            model=None,
            score=0.0,
            runner_up_score=0.0,
            ambiguous=False,
            note=f"可用峰只有 {len(usable)} 个，至少需要 4 个不同能量的峰才能自动刻度。",
        )

    # 只保留最显著的一批峰，既控复杂度也避免本底小起伏参与配对
    ranked = sorted(usable, key=lambda item: item.area * max(item.snr, 1.0), reverse=True)[: int(max_peaks)]
    ranked.sort(key=lambda item: item.channel)
    channels = np.asarray([peak.channel for peak in ranked], dtype=float)
    fwhms = np.asarray([peak.fwhm for peak in ranked], dtype=float)
    median_fwhm = float(np.median(fwhms[fwhms > 0])) if np.any(fwhms > 0) else 4.0
    tolerance_channels = max(1.5, 0.55 * median_fwhm)

    lines: List[Dict[str, float]] = []
    for isotope, entries in reference.items():
        for entry in entries:
            if float(entry["intensity"]) >= float(min_intensity):
                lines.append(
                    {
                        "isotope": isotope,
                        "energy": float(entry["energy"]),
                        "intensity": float(entry["intensity"]),
                    }
                )
    if len(lines) < 4:
        return AutoCalibrationResult(
            model=None,
            score=0.0,
            runner_up_score=0.0,
            ambiguous=False,
            note="当前核素库的有效参考线太少，无法自动刻度。",
        )

    line_energy = np.asarray([line["energy"] for line in lines], dtype=float)
    line_intensity = np.asarray([line["intensity"] for line in lines], dtype=float)
    isotopes = sorted({line["isotope"] for line in lines})
    isotope_index = {name: index for index, name in enumerate(isotopes)}
    line_isotope = np.asarray([isotope_index[line["isotope"]] for line in lines], dtype=int)

    totals = np.zeros(len(isotopes), dtype=float)
    for index, weight in zip(line_isotope, line_intensity):
        totals[index] += weight
    totals = np.maximum(totals, 1e-12)
    one_hot = np.zeros((len(lines), len(isotopes)), dtype=float)
    one_hot[np.arange(len(lines)), line_isotope] = 1.0

    strong = np.where(line_intensity >= float(pair_intensity))[0]
    if len(strong) < 2:
        strong = np.argsort(-line_intensity)[: min(8, len(lines))]
    pair_left, pair_right = [], []
    for i_index in range(len(strong)):
        for j_index in range(i_index + 1, len(strong)):
            first, second = strong[i_index], strong[j_index]
            if line_energy[first] == line_energy[second]:
                continue
            if line_energy[first] < line_energy[second]:
                pair_left.append(first)
                pair_right.append(second)
            else:
                pair_left.append(second)
                pair_right.append(first)
    if not pair_left:
        return AutoCalibrationResult(
            model=None, score=0.0, runner_up_score=0.0, ambiguous=False, note="核素库中没有可用于配对的不同能量线。"
        )
    pair_left = np.asarray(pair_left, dtype=int)
    pair_right = np.asarray(pair_right, dtype=int)
    delta_energy = line_energy[pair_right] - line_energy[pair_left]

    scored: List[Tuple[float, float, float]] = []
    evaluated = 0
    channel_span = float(channels[-1])

    for i_index in range(len(channels) - 1):
        for j_index in range(i_index + 1, len(channels)):
            delta_channel = channels[j_index] - channels[i_index]
            if delta_channel <= 0:
                continue
            slopes = delta_energy / delta_channel
            offsets = line_energy[pair_left] - slopes * channels[i_index]
            keep = (
                (slopes >= float(slope_bounds[0]))
                & (slopes <= float(slope_bounds[1]))
                & (offsets > -50.0)
                & (offsets + slopes * channel_span < 3200.0)
            )
            if not np.any(keep):
                continue
            slopes = slopes[keep]
            offsets = offsets[keep]
            evaluated += int(len(slopes))
            scores = _score_hypotheses(
                slopes,
                offsets,
                channels,
                tolerance_channels,
                line_energy,
                line_intensity,
                line_isotope,
                isotopes,
                totals,
                one_hot,
                absolute_tolerance_kev,
                relative_tolerance,
                min_points,
            )
            for index in np.argsort(-scores)[:3]:
                scored.append((float(scores[index]), float(slopes[index]), float(offsets[index])))

    if not scored:
        return AutoCalibrationResult(
            model=None,
            score=0.0,
            runner_up_score=0.0,
            ambiguous=False,
            note="在给定的斜率范围内没有产生任何候选刻度。",
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    if best_score <= 0:
        return AutoCalibrationResult(
            model=None,
            score=0.0,
            runner_up_score=0.0,
            ambiguous=False,
            hypotheses=evaluated,
            note="所有候选刻度都无法让任何参考线落在实测峰上，请先确认核素库是否匹配该样品。",
        )

    # 对排名前的候选逐个精修，再排序：两点评分只是初筛，真正可比的是精修后的分数
    refined_candidates: List[Tuple[float, float, float, List[Tuple[int, float, float]]]] = []
    scanned = 0
    for raw_score, slope, offset in scored:
        if scanned >= 60 or len(refined_candidates) >= 8:
            break
        scanned += 1
        if any(abs(slope - value[1]) / max(abs(value[1]), 1e-9) < 1e-3 for value in refined_candidates):
            continue
        refined = _refine_hypothesis(
            slope,
            offset,
            channels,
            tolerance_channels,
            line_energy,
            line_intensity,
            line_isotope,
            isotopes,
            totals,
            one_hot,
            absolute_tolerance_kev,
            relative_tolerance,
            min_points,
        )
        if refined is None:
            continue
        (refined_slope, refined_offset), refined_score, entries = refined
        if refined_score <= 0:
            continue
        # 同一组峰上的同一套参考线就是同一个解，即使最小二乘收敛到略微不同的斜率
        if any(_assignment_overlap(entries, value[3]) for value in refined_candidates):
            continue
        refined_candidates.append((refined_score, refined_slope, refined_offset, entries))

    if not refined_candidates:
        return AutoCalibrationResult(
            model=None,
            score=best_score,
            runner_up_score=0.0,
            ambiguous=False,
            hypotheses=evaluated,
            note="最优候选无法完成最小二乘精修。",
        )

    refined_candidates.sort(key=lambda item: item[0], reverse=True)
    score, slope, offset, entries = refined_candidates[0]
    runner_up = float(refined_candidates[1][0]) if len(refined_candidates) > 1 else 0.0
    model = CalibrationModel([float(slope), float(offset)])

    matched = []
    for entry in entries:
        matched.append(
            CalibrationMatch(
                isotope=isotopes[int(line_isotope[entry[0]])],
                energy=float(line_energy[entry[0]]),
                intensity=float(line_intensity[entry[0]]),
                channel=float(entry[1]),
                residual_kev=float(line_energy[entry[0]] - model.energy(entry[1])),
                quality=float(entry[2]),
            )
        )
    counts: Dict[str, int] = {}
    for item in matched:
        counts[item.isotope] = counts.get(item.isotope, 0) + 1
    ranking = sorted(
        ((isotope, counts[isotope], intensity_share(isotope, matched, reference)) for isotope in counts),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )

    # 归一化残差：真实解残差远小于容差；偶然对齐的残差在容差内近似均匀分布
    normalized_residuals = []
    for item in matched:
        local_tolerance = max(
            float(absolute_tolerance_kev),
            item.energy * float(relative_tolerance),
            tolerance_channels * abs(slope),
        )
        normalized_residuals.append(abs(item.residual_kev) / max(local_tolerance, 1e-9))
    normalized_rms = (
        float(np.sqrt(np.mean(np.square(normalized_residuals)))) if normalized_residuals else 1.0
    )
    residual_rms_kev = (
        float(np.sqrt(np.mean(np.square([item.residual_kev for item in matched])))) if matched else 0.0
    )

    ambiguous = runner_up > 0 and score > 0 and runner_up / score > 0.95
    # 命中线数必须明显多于拟合参数（2 个），否则残差小只是过拟合的假象
    credible = (not ambiguous) and len(matched) >= 5 and normalized_rms < 0.35
    note = ""
    if ambiguous:
        note = (
            f"存在近似同分的第二个解（{runner_up / score * 100:.0f}%），"
            "无法唯一确定刻度，建议补充标准源谱线或改用两点手工刻度。"
        )
    elif len(matched) < 5:
        note = "命中参考线少于 5 条，拟合参数几乎与数据点同量，刻度只能视为初筛结果。"
    elif normalized_rms >= 0.35:
        note = f"归一化残差 {normalized_rms:.2f} 偏大，命中可能来自偶然重合，请人工核对。"

    return AutoCalibrationResult(
        model=model,
        score=float(score),
        runner_up_score=float(runner_up),
        ambiguous=ambiguous,
        matched=matched,
        nuclide_ranking=ranking,
        hypotheses=evaluated,
        note=note,
        residual_rms_kev=residual_rms_kev,
        normalized_rms=normalized_rms,
        credible=bool(credible),
    )


def intensity_share(isotope: str, matched: List[CalibrationMatch], reference: Dict[str, List[Dict[str, float]]]) -> float:
    lines = reference.get(isotope, [])
    total = sum(float(line["intensity"]) for line in lines) or 1.0
    hit = sum(item.intensity for item in matched if item.isotope == isotope)
    return hit / total


def _assignment_overlap(
    first: List[Tuple[int, float, float]],
    second: List[Tuple[int, float, float]],
    threshold: float = 0.7,
) -> bool:
    """True when two hypotheses put reference lines on essentially the same peaks."""

    peaks_first = {round(entry[1], 1) for entry in first}
    peaks_second = {round(entry[1], 1) for entry in second}
    if not peaks_first or not peaks_second:
        return False
    shared = len(peaks_first & peaks_second)
    return shared / min(len(peaks_first), len(peaks_second)) >= threshold


def _score_hypotheses(
    slopes: np.ndarray,
    offsets: np.ndarray,
    channels: np.ndarray,
    tolerance_channels: float,
    line_energy: np.ndarray,
    line_intensity: np.ndarray,
    line_isotope: np.ndarray,
    isotopes: List[str],
    totals: np.ndarray,
    one_hot: np.ndarray,
    absolute_tolerance_kev: float,
    relative_tolerance: float,
    min_points: int,
) -> np.ndarray:
    """Vectorised global vote for a batch of linear calibration hypotheses."""

    predicted = (line_energy[None, :] - offsets[:, None]) / slopes[:, None]
    distance = _nearest_channel_distance(predicted, channels)
    energy_tolerance = np.maximum(absolute_tolerance_kev, line_energy[None, :] * relative_tolerance)
    in_channel = distance <= tolerance_channels
    in_energy = distance * slopes[:, None] <= energy_tolerance
    valid = in_channel & in_energy
    closeness = np.clip(1.0 - distance / tolerance_channels, 0.0, 1.0)
    weighted = np.where(valid, line_intensity[None, :] * closeness, 0.0)

    matched_intensity = weighted @ one_hot
    matched_counts = valid.astype(float) @ one_hot
    coverage = matched_intensity / totals[None, :]
    # 强度加权的平均吻合度：弱线不应该拉低强线的结论
    quality = np.divide(
        matched_intensity,
        np.maximum(np.where(valid, line_intensity[None, :], 0.0) @ one_hot, 1e-12),
    )

    reward = np.where(
        matched_counts >= 3.0,
        1.15,
        np.where(matched_counts >= 2.0, 1.0, np.where(matched_counts >= 1.0, 0.55, 0.0)),
    )
    per_isotope = coverage * reward * (0.7 + 0.3 * np.clip(quality, 0.0, 1.0))
    scores = per_isotope.sum(axis=1)
    scores[valid.sum(axis=1) < int(min_points)] = 0.0
    return scores


def _refine_hypothesis(
    slope: float,
    offset: float,
    channels: np.ndarray,
    tolerance_channels: float,
    line_energy: np.ndarray,
    line_intensity: np.ndarray,
    line_isotope: np.ndarray,
    isotopes: List[str],
    totals: np.ndarray,
    one_hot: np.ndarray,
    absolute_tolerance_kev: float,
    relative_tolerance: float,
    min_points: int,
) -> Optional[Tuple[Tuple[float, float], float, List[Tuple[int, float, float]]]]:
    """Least-squares refinement of one hypothesis on its matched points."""

    best: Optional[Tuple[Tuple[float, float], float, List[Tuple[int, float, float]]]] = None
    current_slope, current_offset = slope, offset
    for _ in range(5):
        predicted = (line_energy - current_offset) / current_slope
        distance = _nearest_channel_distance(predicted, channels)
        energy_tolerance = np.maximum(absolute_tolerance_kev, line_energy * relative_tolerance)
        valid = (distance <= tolerance_channels) & (distance * current_slope <= energy_tolerance)
        if int(valid.sum()) < int(min_points):
            break
        entries = [
            (
                int(index),
                float(channels[int(np.argmin(np.abs(channels - predicted[index])))]),
                float(max(0.0, 1.0 - distance[index] / tolerance_channels)),
            )
            for index in np.where(valid)[0]
        ]
        # 同一条实测峰只能贡献一个拟合点：双线情形下两条参考线共用峰心，
        # 若都参与拟合，等于把同一个质心当成两个独立测量，会系统拉偏刻度
        best_per_channel: Dict[float, Tuple[int, float, float]] = {}
        for entry in entries:
            key = round(entry[1], 3)
            if key not in best_per_channel or entry[2] > best_per_channel[key][2]:
                best_per_channel[key] = entry
        fit_points = list(best_per_channel.values())
        x = np.asarray([entry[1] for entry in fit_points], dtype=float)
        y = np.asarray([line_energy[entry[0]] for entry in fit_points], dtype=float)
        weight = np.asarray([line_intensity[entry[0]] for entry in fit_points], dtype=float)
        if len(np.unique(x)) < 2:
            break
        coeffs = np.polyfit(x, y, 1, w=np.sqrt(np.maximum(weight, 1e-6)))
        new_slope, new_offset = float(coeffs[0]), float(coeffs[1])
        if new_slope <= 0:
            break
        current_slope, current_offset = new_slope, new_offset
        score = float(
            _score_hypotheses(
                np.asarray([current_slope]),
                np.asarray([current_offset]),
                channels,
                tolerance_channels,
                line_energy,
                line_intensity,
                line_isotope,
                isotopes,
                totals,
                one_hot,
                absolute_tolerance_kev,
                relative_tolerance,
                min_points,
            )[0]
        )
        if best is None or score >= best[1]:
            best = ((current_slope, current_offset), score, entries)
        else:
            break
    return best


class EfficiencyModel:
    """对 ln(ε) 与 ln(E) 做多项式拟合的效率刻度模型。"""

    def __init__(
        self,
        coefficients: Optional[List[float]] = None,
        energy_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.coefficients = coefficients or []
        self.energy_bounds = energy_bounds

    def is_valid(self) -> bool:
        return bool(self.coefficients)

    def set_energy_bounds(self, energies: List[float]) -> None:
        valid = [value for value in energies if value > 0]
        if not valid:
            self.energy_bounds = None
            return
        low = min(valid)
        high = max(valid)
        if math.isclose(low, high):
            margin = max(5.0, low * 0.05)
        else:
            span = max(high - low, 1.0)
            margin = max(5.0, span * 0.05)
        self.energy_bounds = (max(1e-3, low - margin), high + margin)

    def within_domain(self, energy: float) -> bool:
        if energy <= 0:
            return False
        if not self.energy_bounds:
            return True
        low, high = self.energy_bounds
        return low <= energy <= high

    def efficiency(self, energy: float) -> float:
        if not self.coefficients or energy <= 0 or not self.within_domain(energy):
            return 0.0
        log_energy = math.log(energy)
        log_eff = 0.0
        degree = len(self.coefficients) - 1
        for idx, coef in enumerate(self.coefficients):
            power = degree - idx
            log_eff += coef * (log_energy ** power)
        return math.exp(log_eff)

    def describe(self) -> str:
        if not self.coefficients:
            return "未刻度"
        terms = []
        degree = len(self.coefficients) - 1
        for idx, coef in enumerate(self.coefficients):
            power = degree - idx
            if power == 0:
                terms.append(f"{coef:.6f}")
            elif power == 1:
                terms.append(f"{coef:.6f}·lnE")
            else:
                terms.append(f"{coef:.6f}·lnE^{power}")
        return "lnε = " + " + ".join(terms)


def _validated_record(record: SpectrumRecord) -> SpectrumRecord:
    """Normalize arrays and reject data that would make analysis misleading."""

    channels = np.asarray(record.channels, dtype=float)
    counts = np.asarray(record.counts, dtype=float)
    if channels.ndim != 1 or counts.ndim != 1 or len(channels) != len(counts):
        raise ValueError("通道与计数必须是一维且长度一致。")
    if len(channels) < 4:
        raise ValueError("谱数据点不足，至少需要 4 个通道。")
    if not np.all(np.isfinite(channels)) or not np.all(np.isfinite(counts)):
        raise ValueError("谱数据包含 NaN 或无穷值。")
    if np.any(counts < 0):
        raise ValueError("谱计数不能为负数。")
    order = np.argsort(channels)
    channels = channels[order]
    counts = counts[order]
    if np.any(np.diff(channels) <= 0):
        raise ValueError("通道编号必须严格递增且不能重复。")
    return SpectrumRecord(
        filepath=record.filepath,
        channels=channels,
        counts=counts,
        real_time=record.real_time,
        live_time=record.live_time,
        warnings=list(record.warnings),
        measurement_date=record.measurement_date,
    )


def _iso_date_from_text(text: str) -> Optional[str]:
    """Parse a measurement date such as ``11/11/2022 22:10:26`` into ISO form.

    GammaVision writes US-style ``MM/DD/YYYY``; Chinese instruments usually write
    ``YYYY-MM-DD`` or ``YYYY年M月D日``. Both orders are accepted here.
    """

    if not text:
        return None
    match = re.search(r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
    else:
        match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
        if not match:
            return None
        month, day, year = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_spe_file(filepath: str) -> SpectrumRecord:
    counts: List[float] = []
    start_channel = 0
    end_channel: Optional[int] = None
    expected_channels: Optional[int] = None
    reading_data = False
    pending_tag: Optional[str] = None
    real_time = live_time = None
    measurement_date: Optional[str] = None

    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if reading_data and not line.startswith("$"):
                try:
                    counts.append(float(line))
                except ValueError:
                    continue
                continue

            if pending_tag == "$DATA_RANGE" and not line.startswith("$"):
                parts = line.replace(",", " ").split()
                try:
                    if len(parts) < 2:
                        raise ValueError
                    start_channel = int(float(parts[0]))
                    end_channel = int(float(parts[1]))
                except ValueError as exc:
                    raise ValueError("SPE 的 $DATA 区间无效。") from exc
                if end_channel < start_channel:
                    raise ValueError("SPE 的 $DATA 结束通道小于起始通道。")
                expected_channels = end_channel - start_channel + 1
                pending_tag = None
                reading_data = True
                continue

            if pending_tag == "$MEAS_TIM" and not line.startswith("$"):
                values = line.replace(",", " ").split()
                try:
                    if len(values) >= 2:
                        # SPE convention: live time first, real time second.
                        live_time = float(values[0])
                        real_time = float(values[1])
                except ValueError:
                    pass
                pending_tag = None
                continue

            if pending_tag == "$DATE_MEA" and not line.startswith("$"):
                measurement_date = _iso_date_from_text(line) or measurement_date
                pending_tag = None
                continue

            if line.startswith("$"):
                reading_data = False
                tag, _, payload = line.partition(":")
                upper_tag = tag.upper()
                if upper_tag.startswith("$DATA"):
                    parts = payload.replace(",", " ").split()
                    if len(parts) >= 2:
                        try:
                            start_channel = int(float(parts[0]))
                            end_channel = int(float(parts[1]))
                        except ValueError as exc:
                            raise ValueError("SPE 的 $DATA 区间无效。") from exc
                        if end_channel < start_channel:
                            raise ValueError("SPE 的 $DATA 结束通道小于起始通道。")
                        expected_channels = end_channel - start_channel + 1
                        reading_data = True
                        pending_tag = None
                    else:
                        pending_tag = "$DATA_RANGE"
                elif upper_tag.startswith("$MEAS_TIM"):
                    values = payload.replace(",", " ").split()
                    if len(values) >= 2:
                        try:
                            live_time = float(values[0])
                            real_time = float(values[1])
                        except ValueError:
                            pass
                        pending_tag = None
                    else:
                        pending_tag = "$MEAS_TIM"
                elif upper_tag.startswith("$DATE_MEA"):
                    measurement_date = _iso_date_from_text(payload) or measurement_date
                    pending_tag = None if measurement_date else "$DATE_MEA"
                else:
                    pending_tag = None
                continue

    warnings: List[str] = []
    if expected_channels is not None and len(counts) != expected_channels:
        exclusive_count = (end_channel - start_channel) if end_channel is not None else -1
        if len(counts) == exclusive_count:
            warnings.append(
                "SPE 的 $DATA 结束值按上界解释；文件实际点数比闭区间声明少 1。"
            )
        else:
            raise ValueError(
                f"SPE 数据长度不匹配：闭区间声明 {expected_channels} 个通道，实际读取 {len(counts)} 个。"
            )
    if not counts:
        raise ValueError("SPE 文件中未找到有效计数数据。")

    channels = np.arange(start_channel, start_channel + len(counts))
    return _validated_record(SpectrumRecord(
        filepath=filepath,
        channels=channels,
        counts=np.asarray(counts, dtype=float),
        real_time=real_time,
        live_time=live_time,
        warnings=warnings,
        measurement_date=measurement_date,
    ))


def _normalize_header_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s_\-:/\\()\[\]{}（）【】]+", "", text)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _header_matches(label: object, keywords: Tuple[str, ...]) -> bool:
    normalized = _normalize_header_label(label)
    return bool(normalized) and any(keyword in normalized for keyword in keywords)


def _xlsx_column_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return max(index - 1, 0)


def _xlsx_inline_text(cell: ET.Element) -> str:
    parts = [node.text or "" for node in cell.findall(f".//{{{XLSX_MAIN_NS}}}t")]
    return "".join(parts).strip()


def _read_xlsx_shared_strings(workbook: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read(path))
    values: List[str] = []
    for item in root.findall(f".//{{{XLSX_MAIN_NS}}}si"):
        parts = [node.text or "" for node in item.findall(f".//{{{XLSX_MAIN_NS}}}t")]
        values.append("".join(parts))
    return values


def _resolve_xlsx_target(base_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath(base_path).parent.joinpath(target))


def _read_xlsx_sheet_rows(workbook: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]) -> List[Dict[int, str]]:
    root = ET.fromstring(workbook.read(sheet_path))
    rows: List[Dict[int, str]] = []
    for row in root.findall(f".//{{{XLSX_MAIN_NS}}}sheetData/{{{XLSX_MAIN_NS}}}row"):
        values: Dict[int, str] = {}
        for cell in row.findall(f"{{{XLSX_MAIN_NS}}}c"):
            ref = cell.get("r", "")
            column = _xlsx_column_to_index(ref)
            cell_type = cell.get("t", "")
            if cell_type == "inlineStr":
                text = _xlsx_inline_text(cell)
            else:
                raw_value = cell.findtext(f"{{{XLSX_MAIN_NS}}}v", default="")
                if cell_type == "s":
                    index = int(raw_value) if raw_value.isdigit() else -1
                    text = shared_strings[index] if 0 <= index < len(shared_strings) else ""
                elif cell_type == "b":
                    text = "1" if raw_value == "1" else "0"
                else:
                    text = raw_value
            if str(text).strip():
                values[column] = str(text).strip()
        rows.append(values)
    return rows


def _extract_time_from_row(row_values: List[object]) -> Tuple[Optional[float], Optional[float]]:
    real_time = None
    live_time = None
    for idx in range(len(row_values) - 1):
        label = row_values[idx]
        value = _safe_float(row_values[idx + 1])
        if value is None:
            continue
        if real_time is None and _header_matches(label, REAL_TIME_HEADER_KEYWORDS):
            real_time = value
        if live_time is None and _header_matches(label, LIVE_TIME_HEADER_KEYWORDS):
            live_time = value
    return real_time, live_time


def _best_numeric_run(rows: List[Dict[int, str]], counts_col: int, channel_col: Optional[int]) -> Tuple[int, int]:
    best_start = -1
    best_length = 0
    current_start = -1
    current_length = 0
    for idx, row in enumerate(rows):
        count_value = _safe_float(row.get(counts_col))
        channel_value = _safe_float(row.get(channel_col)) if channel_col is not None else None
        valid = count_value is not None and (channel_col is None or channel_value is not None)
        if valid:
            if current_start < 0:
                current_start = idx
                current_length = 0
            current_length += 1
        else:
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
            current_start = -1
            current_length = 0
    if current_length > best_length:
        best_start = current_start
        best_length = current_length
    return best_start, best_length


def _parse_xlsx_sheet(filepath: str, sheet_name: str, rows: List[Dict[int, str]]) -> Optional[SpectrumRecord]:
    if not rows:
        return None

    header_row_index = -1
    counts_col: Optional[int] = None
    channel_col: Optional[int] = None
    real_time_col: Optional[int] = None
    live_time_col: Optional[int] = None
    real_time: Optional[float] = None
    live_time: Optional[float] = None

    for idx, row in enumerate(rows[:20]):
        ordered_values = [row[key] for key in sorted(row)]
        row_real, row_live = _extract_time_from_row(ordered_values)
        if real_time is None and row_real is not None:
            real_time = row_real
        if live_time is None and row_live is not None:
            live_time = row_live
        if header_row_index >= 0:
            continue
        if not row:
            continue
        if any(_header_matches(value, COUNT_HEADER_KEYWORDS) for value in row.values()):
            header_row_index = idx
        elif any(_header_matches(value, CHANNEL_HEADER_KEYWORDS) for value in row.values()):
            header_row_index = idx
        if header_row_index >= 0:
            for column, value in row.items():
                if counts_col is None and _header_matches(value, COUNT_HEADER_KEYWORDS):
                    counts_col = column
                if channel_col is None and _header_matches(value, CHANNEL_HEADER_KEYWORDS):
                    channel_col = column
                if real_time_col is None and _header_matches(value, REAL_TIME_HEADER_KEYWORDS):
                    real_time_col = column
                if live_time_col is None and _header_matches(value, LIVE_TIME_HEADER_KEYWORDS):
                    live_time_col = column

    numeric_counts: Dict[int, int] = {}
    for row in rows[max(header_row_index + 1, 0):]:
        for column, value in row.items():
            if _safe_float(value) is not None:
                numeric_counts[column] = numeric_counts.get(column, 0) + 1

    ranked_numeric_columns = [column for column, _ in sorted(numeric_counts.items(), key=lambda item: (-item[1], item[0]))]
    numeric_columns = [column for column, _ in sorted(numeric_counts.items(), key=lambda item: item[0])]
    preferred_numeric_columns = sorted(ranked_numeric_columns[:2])
    if counts_col is None:
        if len(preferred_numeric_columns) == 1:
            counts_col = preferred_numeric_columns[0]
        elif len(preferred_numeric_columns) >= 2:
            channel_col = preferred_numeric_columns[0] if channel_col is None else channel_col
            counts_col = preferred_numeric_columns[1]
        elif len(numeric_columns) == 1:
            counts_col = numeric_columns[0]
        elif len(numeric_columns) >= 2:
            channel_col = numeric_columns[0] if channel_col is None else channel_col
            counts_col = numeric_columns[1]
    elif channel_col is None and numeric_columns:
        for column in numeric_columns:
            if column != counts_col:
                channel_col = column
                break

    if counts_col is None:
        return None

    start_index = max(header_row_index + 1, 0)
    run_start, run_length = _best_numeric_run(rows[start_index:], counts_col, channel_col)
    if run_start < 0 or run_length < 4:
        return None
    run_start += start_index
    run_end = run_start + run_length

    channels: List[float] = []
    counts: List[float] = []
    for row in rows[run_start:run_end]:
        count_value = _safe_float(row.get(counts_col))
        if count_value is None:
            continue
        channel_value = _safe_float(row.get(channel_col)) if channel_col is not None else None
        if channel_col is not None and channel_value is None:
            continue
        counts.append(count_value)
        if channel_value is None:
            channels.append(float(len(channels)))
        else:
            channels.append(channel_value)
        if real_time is None and real_time_col is not None:
            real_time = _safe_float(row.get(real_time_col))
        if live_time is None and live_time_col is not None:
            live_time = _safe_float(row.get(live_time_col))

    if len(counts) < 4:
        return None

    channel_array = np.asarray(channels, dtype=float)
    counts_array = np.asarray(counts, dtype=float)
    order = np.argsort(channel_array)
    channel_array = channel_array[order]
    counts_array = counts_array[order]
    if len(channel_array) >= 2 and np.any(np.diff(channel_array) <= 0):
        channel_array = np.arange(len(counts_array), dtype=float)

    return _validated_record(SpectrumRecord(
        filepath=filepath,
        channels=channel_array,
        counts=counts_array,
        real_time=real_time,
        live_time=live_time,
    ))


def parse_xlsx_file(filepath: str) -> SpectrumRecord:
    try:
        with zipfile.ZipFile(filepath, "r") as workbook:
            shared_strings = _read_xlsx_shared_strings(workbook)
            workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
            rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
            rel_map = {
                rel.get("Id", ""): _resolve_xlsx_target("xl/workbook.xml", rel.get("Target", ""))
                for rel in rel_root.findall(f".//{{{XLSX_PKG_REL_NS}}}Relationship")
            }

            for sheet in workbook_root.findall(f".//{{{XLSX_MAIN_NS}}}sheet"):
                rel_id = sheet.get(f"{{{XLSX_REL_NS}}}id", "")
                sheet_name = sheet.get("name", "Sheet1")
                sheet_path = rel_map.get(rel_id)
                if not sheet_path or sheet_path not in workbook.namelist():
                    continue
                rows = _read_xlsx_sheet_rows(workbook, sheet_path, shared_strings)
                parsed = _parse_xlsx_sheet(filepath, sheet_name, rows)
                if parsed is not None:
                    return parsed
    except (OSError, KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"无法读取 Excel 文件: {exc}") from exc
    raise ValueError("Excel 文件中未找到可识别的光谱数据。")


def parse_delimited_file(filepath: str) -> SpectrumRecord:
    """Read a UTF-8/GB18030 CSV or tab-delimited channel-count file."""

    raw: Optional[str] = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            raw = Path(filepath).read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ValueError(f"无法读取文本谱文件: {exc}") from exc
    if raw is None:
        raise ValueError("文本谱文件编码无法识别，请使用 UTF-8 或 GB18030。")

    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("文本谱文件为空。")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t")
        rows_raw = list(csv.reader(lines, dialect))
    except csv.Error:
        rows_raw = [re.split(r"[,;\t\s]+", line.strip()) for line in lines]

    rows = [
        {index: value.strip() for index, value in enumerate(row) if value.strip()}
        for row in rows_raw
    ]
    parsed = _parse_xlsx_sheet(filepath, Path(filepath).stem, rows)
    if parsed is None:
        raise ValueError("文本文件中未找到可识别的通道-计数数据。")
    return parsed


def load_spectrum_file(filepath: str) -> SpectrumRecord:
    extension = os.path.splitext(filepath)[1].lower()
    if extension == ".spe":
        return parse_spe_file(filepath)
    if extension == ".xlsx":
        return parse_xlsx_file(filepath)
    if extension in {".csv", ".txt"}:
        return parse_delimited_file(filepath)
    raise ValueError("暂不支持该文件类型，请选择 SPE、XLSX、CSV 或 TXT 文件。")


def subtract_background(
    record: SpectrumRecord,
    background: SpectrumRecord,
    scale_mode: str = "live",
    floor: float = 0.0,
) -> Tuple[SpectrumRecord, BackgroundInfo]:
    """Subtract a measured background spectrum from a sample spectrum.

    The background is normalized to the sample by the live-time (or real-time)
    ratio before subtraction, so both spectra are compared on the same counting
    rate basis. A background measured on a different channel grid is linearly
    resampled onto the sample grid. Channels that would go negative are clipped
    to ``floor`` and counted as clipped, because this prototype has no
    uncertainty propagation for negative net counts.
    """

    if scale_mode not in {"live", "real", "none"}:
        raise ValueError("本底归一方式必须是 live、real 或 none。")

    sample = _validated_record(record)
    bg = _validated_record(background)
    warnings: List[str] = []

    same_grid = (
        len(sample.channels) == len(bg.channels)
        and np.allclose(sample.channels, bg.channels, rtol=0.0, atol=1e-6)
    )
    if same_grid:
        bg_counts = bg.counts
        resampled = False
    else:
        if sample.channels[-1] < bg.channels[0] or sample.channels[0] > bg.channels[-1]:
            raise ValueError("本底谱与样品谱的道址范围没有重叠，无法扣除。")
        bg_counts = np.interp(sample.channels, bg.channels, bg.counts, left=0.0, right=0.0)
        resampled = True
        warnings.append(
            f"本底谱道址范围与样品谱不同（{len(bg.channels)} 道 → {len(sample.channels)} 道），已线性重采样。"
        )

    scale = 1.0
    if scale_mode == "live":
        if sample.live_time and bg.live_time and sample.live_time > 0 and bg.live_time > 0:
            scale = float(sample.live_time) / float(bg.live_time)
        elif sample.real_time and bg.real_time and sample.real_time > 0 and bg.real_time > 0:
            scale = float(sample.real_time) / float(bg.real_time)
            warnings.append("本底谱或样品谱缺少活时间，已改用实时间比归一。")
        else:
            warnings.append("缺少计时信息，本底谱按 1:1 直接扣除；若测量时间不同会引入系统偏差。")
    elif scale_mode == "real":
        if sample.real_time and bg.real_time and sample.real_time > 0 and bg.real_time > 0:
            scale = float(sample.real_time) / float(bg.real_time)
        else:
            warnings.append("缺少实时间信息，本底谱按 1:1 直接扣除。")

    subtracted = sample.counts - bg_counts * scale
    clipped = int(np.count_nonzero(subtracted < floor))
    net = np.clip(subtracted, floor, None)
    clipped_fraction = clipped / max(len(net), 1)
    if clipped:
        warnings.append(
            f"扣除后有 {clipped} 个通道 ({clipped_fraction * 100:.1f}%) 低于 {floor:g}，已截断到下限；"
            "该部分统计偏差未传播。"
        )

    info = BackgroundInfo(
        filepath=bg.filepath,
        scale_mode=scale_mode,
        scale=scale,
        resampled=resampled,
        clipped_channels=clipped,
        clipped_fraction=clipped_fraction,
        warnings=warnings,
    )
    result = SpectrumRecord(
        filepath=sample.filepath,
        channels=sample.channels,
        counts=net,
        real_time=sample.real_time,
        live_time=sample.live_time,
        warnings=list(sample.warnings) + warnings,
    )
    return result, info


def snip_baseline(counts: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
    spec = np.log(np.clip(counts, 1, None))
    baseline = spec.copy()
    length = len(spec)
    max_iter = iterations or max(16, min(length // 2, 80))
    for k in range(1, max_iter):
        left = np.roll(baseline, k)
        right = np.roll(baseline, -k)
        avg = 0.5 * (left + right)
        avg[:k] = baseline[:k]
        avg[-k:] = baseline[-k:]
        baseline = np.minimum(baseline, avg)
    return np.exp(baseline)


def smooth_spectrum(counts: np.ndarray, window: int = 7) -> np.ndarray:
    size = max(3, int(window))
    if size % 2 == 0:
        size += 1
    if len(counts) <= 2:
        return np.asarray(counts, dtype=float).copy()
    radius = min(size // 2, max(len(counts) // 2, 1))
    x = np.arange(-radius, radius + 1, dtype=float)
    sigma = max(radius / 2.0, 1.0)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(counts, dtype=float), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def mexican_hat_wavelet(width: float) -> np.ndarray:
    width = max(width, 1.0)
    half = int(4 * width)
    x = np.arange(-half, half + 1)
    sigma = width
    norm = 2 / (math.sqrt(3 * sigma) * (math.pi ** 0.25))
    wave = norm * (1 - (x ** 2) / (sigma ** 2)) * np.exp(-(x ** 2) / (2 * sigma ** 2))
    wave -= wave.mean()
    return wave


def continuous_wavelet_transform(counts: np.ndarray, widths: List[int]) -> np.ndarray:
    responses = []
    for width in widths:
        kernel = mexican_hat_wavelet(float(width))
        conv = np.convolve(counts, kernel, mode="same")
        responses.append(conv)
    return np.asarray(responses)


def detect_peaks_from_cwt(
    net_counts: np.ndarray,
    widths: Optional[List[int] | Tuple[int, ...]] = None,
    min_distance: int = 6,
    threshold_sigma: float = 1.0,
) -> List[int]:
    net_counts = np.asarray(net_counts, dtype=float)
    if net_counts.ndim != 1 or len(net_counts) < 3:
        return []
    if not np.all(np.isfinite(net_counts)):
        raise ValueError("寻峰输入包含 NaN 或无穷值。")
    if widths is None:
        widths = sorted({2, 4, 6, 8, 10, 12})
    cwt_matrix = continuous_wavelet_transform(net_counts, widths)
    aggregate = np.max(cwt_matrix, axis=0)
    threshold = aggregate.mean() + float(threshold_sigma) * aggregate.std()
    candidates: List[int] = []
    last_idx = -min_distance
    for idx in range(1, len(aggregate) - 1):
        if aggregate[idx] > threshold and aggregate[idx] >= aggregate[idx - 1] and aggregate[idx] >= aggregate[idx + 1]:
            if idx - last_idx < min_distance:
                if aggregate[idx] > aggregate[last_idx]:
                    candidates[-1] = idx
                    last_idx = idx
                continue
            candidates.append(idx)
            last_idx = idx
    return candidates


def refine_peaks(
    channels: np.ndarray,
    net_counts: np.ndarray,
    raw_counts: np.ndarray,
    candidates: List[int],
    window: int = 20,
    min_snr: float = 1.25,
) -> List[PeakResult]:
    peaks: List[PeakResult] = []
    for center in candidates:
        left = max(0, center - window)
        right = min(len(net_counts), center + window + 1)
        segment = net_counts[left:right]
        if len(segment) < 5 or np.max(segment) <= 0:
            continue
        x = channels[left:right]
        baseline = max(segment[0], segment[-1])
        y = np.clip(segment - baseline, 0, None)
        if np.sum(y) <= 0:
            continue
        mu = float(np.sum(x * y) / np.sum(y))
        variance = float(np.sum(y * (x - mu) ** 2) / np.sum(y))
        sigma = math.sqrt(max(variance, 1e-6))
        amplitude = float(np.max(y))
        fwhm = 2.3548 * sigma
        reconstructed = amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        residual = math.sqrt(np.mean((y - reconstructed) ** 2))
        snr = amplitude / max(residual, 1e-6)
        if snr < min_snr:
            continue
        area = float(_trapz(y, x))
        area_uncertainty = 0.0
        fitted = fit_peak_shape(x, segment, mu, sigma, amplitude)
        if fitted is not None:
            fit_area = fitted["area"]
            plausible = (
                0.2 * area <= fit_area <= 5.0 * area
                and 0.5 * sigma <= fitted["sigma"] <= 2.0 * sigma
                and abs(fitted["center"] - mu) <= 1.5 * sigma
            )
            if plausible:
                mu = fitted["center"]
                sigma = fitted["sigma"]
                fwhm = fitted["fwhm"]
                area = fit_area
                area_uncertainty = fitted["area_uncertainty"]
        peaks.append(
            PeakResult(
                channel=mu,
                fwhm=fwhm,
                area=area,
                snr=snr,
                area_uncertainty=area_uncertainty,
            )
        )
    return peaks


def merge_close_peaks(peaks: List[PeakResult], factor: float = 0.5) -> List[PeakResult]:
    """Merge peaks that landed closer than half a line width of each other.

    A broad low-energy line often produces two wavelet maxima a few channels
    apart; after individual fits they can collapse onto the same centroid and
    would otherwise be reported twice. Merged peaks keep the area-weighted centre
    and the summed area, with uncertainties added in quadrature.
    """

    if not peaks:
        return peaks
    ordered = sorted(peaks, key=lambda item: item.channel)
    merged: List[PeakResult] = []
    for peak in ordered:
        if not merged:
            merged.append(peak)
            continue
        previous = merged[-1]
        separation = peak.channel - previous.channel
        threshold = factor * max(peak.fwhm, previous.fwhm)
        if separation <= threshold:
            total_area = previous.area + peak.area
            if total_area > 0:
                previous.channel = (previous.channel * previous.area + peak.channel * peak.area) / total_area
            else:
                previous.channel = 0.5 * (previous.channel + peak.channel)
            previous.fwhm = max(previous.fwhm, peak.fwhm)
            previous.area = total_area
            previous.area_uncertainty = math.hypot(previous.area_uncertainty, peak.area_uncertainty)
            previous.snr = max(previous.snr, peak.snr)
        else:
            merged.append(peak)
    return merged


def _gauss_shape(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)


def fit_peak_shape(
    x: np.ndarray,
    y: np.ndarray,
    mu0: float,
    sigma0: float,
    amplitude0: float,
    max_iter: int = 60,
) -> Optional[Dict[str, float]]:
    """Fit a Gaussian plus a linear baseline to one peak window.

    Returns the centre, width, area, area uncertainty and reduced chi-square, or
    ``None`` when the fit does not converge to a plausible shape. Uncertainties
    come from the linearised covariance matrix and therefore describe counting
    scatter of this model only, not systematic effects such as summing,
    attenuation or an incorrect efficiency curve.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 8 or len(x) != len(y) or not np.all(np.isfinite(y)):
        return None
    x0 = float(np.mean(x))
    p = np.array(
        [max(float(amplitude0), 1.0), float(mu0), max(float(sigma0), 0.5), float(np.min(y)), 0.0],
        dtype=float,
    )

    def model(params: np.ndarray) -> np.ndarray:
        amplitude, mu, sigma, b0, b1 = params
        return amplitude * _gauss_shape(x, mu, sigma) + b0 + b1 * (x - x0)

    def jacobian(params: np.ndarray) -> np.ndarray:
        amplitude, mu, sigma, b0, b1 = params
        shape = _gauss_shape(x, mu, sigma)
        delta = x - mu
        return np.column_stack(
            [
                shape,
                amplitude * shape * delta / sigma ** 2,
                amplitude * shape * delta ** 2 / sigma ** 3,
                np.ones_like(x),
                x - x0,
            ]
        )

    residual = y - model(p)
    rss = float(residual @ residual)
    best_p, best_rss = p.copy(), rss
    lam = 1e-2
    step = np.zeros_like(p)
    for _ in range(max_iter):
        try:
            jac = jacobian(p)
        except (OverflowError, ZeroDivisionError):
            break
        normal = jac.T @ jac
        gradient = jac.T @ residual
        accepted = False
        for _ in range(10):
            try:
                step = np.linalg.solve(normal + lam * np.diag(np.diag(normal) + 1e-12), gradient)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial = p + step
            if trial[2] <= 0.25 or trial[0] <= 0:
                lam *= 10.0
                continue
            trial_residual = y - model(trial)
            trial_rss = float(trial_residual @ trial_residual)
            if np.isfinite(trial_rss) and trial_rss < rss:
                p, residual, rss = trial, trial_residual, trial_rss
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        if rss < best_rss:
            best_p, best_rss = p.copy(), rss
        if abs(step[1]) < 1e-4 and abs(step[2]) < 1e-4 and abs(step[0]) < 1e-3 * max(best_p[0], 1.0):
            break

    amplitude, mu, sigma, _, _ = best_p
    if not np.isfinite(best_rss) or amplitude <= 0 or sigma <= 0:
        return None
    try:
        jac = jacobian(best_p)
        dof = max(len(x) - 5, 1)
        cov = (best_rss / dof) * np.linalg.pinv(jac.T @ jac)
    except (np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
        return None
    root_two_pi = math.sqrt(2.0 * math.pi)
    area = amplitude * sigma * root_two_pi
    gradient_area = np.array([sigma * root_two_pi, 0.0, amplitude * root_two_pi, 0.0, 0.0])
    area_variance = float(gradient_area @ cov @ gradient_area)
    return {
        "center": float(mu),
        "sigma": float(sigma),
        "fwhm": 2.35482 * float(sigma),
        "amplitude": float(amplitude),
        "area": float(area),
        "area_uncertainty": math.sqrt(max(area_variance, 0.0)),
        "reduced_chi2": float(best_rss / dof),
    }


def energy_to_channel(
    calibration: CalibrationModel,
    energy: float,
    channel_min: float,
    channel_max: float,
) -> float:
    """Invert a linear or quadratic calibration by bisection inside a bracket."""

    low, high = float(channel_min), float(channel_max)
    f_low = calibration.energy(low) - float(energy)
    f_high = calibration.energy(high) - float(energy)
    if abs(f_low) < 1e-9:
        return low
    if abs(f_high) < 1e-9:
        return high
    if f_low * f_high > 0:
        slope = calibration.slope(0.5 * (low + high))
        if abs(slope) < 1e-12:
            return 0.5 * (low + high)
        return low + (float(energy) - calibration.energy(low)) / slope
    for _ in range(80):
        middle = 0.5 * (low + high)
        f_middle = calibration.energy(middle) - float(energy)
        if f_low * f_middle <= 0:
            high = middle
        else:
            low, f_low = middle, f_middle
    return 0.5 * (low + high)


def _fit_component_amplitudes(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Linear least squares for fixed centroids and a shared width."""

    x0 = float(np.mean(x))
    columns = [_gauss_shape(x, center, sigma) for center in centers]
    columns.append(np.ones_like(x))
    columns.append(x - x0)
    design = np.column_stack(columns)
    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(design.shape[1]), np.zeros((design.shape[1], design.shape[1])), float("inf")
    residual = y - design @ coefficients
    rss = float(residual @ residual)
    dof = max(len(x) - design.shape[1], 1)
    try:
        cov = (rss / dof) * np.linalg.pinv(design.T @ design)
    except np.linalg.LinAlgError:
        cov = np.zeros((design.shape[1], design.shape[1]))
    return coefficients, cov, rss


def decompose_interference(
    peaks: List[PeakResult],
    channels: np.ndarray,
    net_counts: np.ndarray,
    calibration: CalibrationModel,
    window_sigmas: float = 4.0,
    min_points: int = 10,
    min_separation_fwhm: float = 1.0,
) -> None:
    """Split an unresolved multiplet into library-constrained components.

    Component energies are taken from the reference library, so only the
    amplitudes and one shared width are fitted; with fixed centroids the problem
    is linear in the amplitudes and stays stable. Two gates decide whether the
    result may be trusted:

    * the components must be separated by at least ``min_separation_fwhm`` line
      widths — below that the fit is numerically possible but depends on the peak
      shape, calibration and counting statistics more than the data can support;
    * every component needs a positive area, a relative uncertainty below 50%
      and fitted amplitudes that are not almost perfectly correlated.

    Statistical uncertainty alone is never enough here: with good statistics the
    covariance matrix will happily return a tight 50/50 split of one blended peak.
    """

    channels = np.asarray(channels, dtype=float)
    net_counts = np.asarray(net_counts, dtype=float)
    if len(channels) != len(net_counts):
        return
    for peak in peaks:
        peak.components = []
        peak.decomposition_resolved = False
        peak.decomposition_note = ""
        if not peak.interference:
            continue
        energies = sorted(line.energy for line in peak.interference)
        separations = [
            energies[index + 1] - energies[index] for index in range(len(energies) - 1)
        ]
        closest = min(separations) if separations else float("inf")
        fwhm_kev = peak.fwhm_kev if peak.fwhm_kev > 0 else float(peak.fwhm) * abs(calibration.slope(peak.channel))
        separation_ratio = closest / fwhm_kev if fwhm_kev > 0 else float("inf")
        sigma0 = max(float(peak.fwhm) / 2.35482, 0.5)
        half_window = max(window_sigmas * sigma0, 8.0)
        left = int(np.searchsorted(channels, peak.channel - half_window))
        right = int(np.searchsorted(channels, peak.channel + half_window))
        left = max(left, 0)
        right = min(max(right, left + min_points), len(channels))
        if right - left < min_points:
            continue
        x = channels[left:right]
        y = net_counts[left:right]
        if not np.all(np.isfinite(y)) or float(np.max(y)) <= 0:
            continue
        centers = [
            energy_to_channel(calibration, line.energy, float(x[0]), float(x[-1]))
            for line in peak.interference
        ]
        best: Optional[Tuple[float, float, np.ndarray, np.ndarray]] = None
        for factor in np.linspace(0.45, 1.75, 27):
            sigma = sigma0 * float(factor)
            coefficients, cov, rss = _fit_component_amplitudes(x, y, centers, sigma)
            if best is None or rss < best[0]:
                best = (rss, sigma, coefficients, cov)
        if best is None:
            continue
        _, sigma, coefficients, cov = best
        root_two_pi = math.sqrt(2.0 * math.pi)
        areas = [max(float(coefficients[index]), 0.0) * sigma * root_two_pi for index in range(len(centers))]
        total_area = sum(areas)
        if total_area <= 0:
            peak.decomposition_note = "多峰拟合没有给出正面积，无法分解。"
            continue
        components: List[PeakComponent] = []
        resolved = True
        for index, line in enumerate(peak.interference):
            area = areas[index]
            area_sigma = math.sqrt(max(float(cov[index, index]), 0.0)) * sigma * root_two_pi
            components.append(
                PeakComponent(
                    isotope=line.isotope,
                    energy=line.energy,
                    area=area,
                    area_uncertainty=area_sigma,
                    share=area / total_area,
                    share_uncertainty=area_sigma / total_area,
                )
            )
            if area <= 0 or area_sigma / area >= 0.5:
                resolved = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                denominator = math.sqrt(max(float(cov[i, i]), 0.0) * max(float(cov[j, j]), 0.0))
                if denominator > 0 and abs(float(cov[i, j]) / denominator) >= 0.95:
                    resolved = False
        if separation_ratio < float(min_separation_fwhm):
            resolved = False
        peak.components = components
        peak.decomposition_resolved = resolved
        if resolved:
            peak.decomposition_note = "多峰拟合可分解。"
        elif separation_ratio < float(min_separation_fwhm):
            peak.decomposition_note = (
                f"成分间距仅 {separation_ratio:.2f} FWHM（阈值 {min_separation_fwhm:.2f}），"
                "低于可分解门限；拟合值仅供参考，不作为定量依据。"
            )
        else:
            peak.decomposition_note = (
                "多峰拟合不唯一（成分强相关或相对不确定度 ≥50%），不采信分解值。"
            )


def flatten_library(library: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, float]]:
    flat: List[Dict[str, float]] = []
    for isotope, lines in library.items():
        for line in lines:
            entry = dict(line)
            entry["isotope"] = isotope
            flat.append(entry)
    return flat


def match_nuclides(
    peaks: List[PeakResult],
    calibration: CalibrationModel,
    library: List[Dict[str, float]],
    absolute_tolerance_kev: float = 0.5,
    relative_tolerance: float = 0.002,
    max_candidates: int = 3,
    width_tolerance_factor: float = 0.5,
    interference_width_factor: float = 1.0,
) -> None:
    """Match peaks against reference lines and flag unresolved doublets.

    A tolerance floor of ``width_tolerance_factor * FWHM`` is applied on top of the
    fixed keV/relative tolerances so that a well-resolved peak on a coarse detector
    is not rejected for being a fraction of a line width away from the tabulated
    energy. When two reference lines from different nuclides fall inside
    ``interference_width_factor * FWHM`` around the same peak, the peak is marked
    as an unresolved doublet: no single primary isotope is asserted, because doing
    so would attribute a blended centroid to one nuclide.
    """

    if not peaks:
        return
    for peak in peaks:
        energy = calibration.energy(peak.channel)
        peak.energy = energy
        peak.primary_isotope = ""
        peak.reference_intensity = 0.0
        peak.confidence = 0.0
        peak.interference = []
        peak.fwhm_kev = abs(float(peak.fwhm)) * abs(calibration.slope(peak.channel))
        tolerance = max(
            float(absolute_tolerance_kev),
            energy * float(relative_tolerance),
            float(width_tolerance_factor) * peak.fwhm_kev,
        )
        matches = []
        for entry in library:
            diff = abs(entry["energy"] - energy)
            if diff <= tolerance:
                intensity = entry.get("intensity", 0.0)
                norm_diff = diff / tolerance
                matches.append((norm_diff, -intensity, entry, intensity))
        matches.sort(key=lambda item: (item[0], item[1]))
        peak.nuclide_candidates = [
            f"{entry['isotope']} ({entry['energy']:.1f} keV)"
            for _, __, entry, ___ in matches[: int(max_candidates)]
        ]
        if not matches:
            continue
        norm_diff, _, best_entry, intensity = matches[0]
        peak.reference_intensity = float(intensity)
        confidence = max(0.0, 1.0 - norm_diff)
        confidence *= 0.6 + 0.4 * min(max(intensity, 0.0), 1.0)
        peak.confidence = min(confidence, 1.0)

        window = max(float(interference_width_factor) * peak.fwhm_kev, 1e-9)
        near = [item for item in matches if abs(item[2]["energy"] - best_entry["energy"]) <= window]
        if len({item[2]["isotope"] for item in near}) > 1:
            total_intensity = sum(max(item[3], 0.0) for item in near) or 1.0
            peak.interference = [
                InterferenceLine(
                    isotope=str(item[2]["isotope"]),
                    energy=float(item[2]["energy"]),
                    intensity=float(item[3]),
                    delta_kev=float(item[2]["energy"]) - energy,
                    weight=max(item[3], 0.0) / total_intensity,
                )
                for item in sorted(near, key=lambda item: item[2]["energy"])
            ]
            peak.primary_isotope = ""
            peak.reference_intensity = 0.0
        else:
            peak.primary_isotope = str(best_entry.get("isotope", ""))


def summarize_nuclide_matches(
    peaks: List[PeakResult],
    library: Dict[str, List[Dict[str, float]]],
    absolute_tolerance_kev: float = 0.5,
    relative_tolerance: float = 0.002,
    width_tolerance_factor: float = 0.5,
) -> List[NuclideMatchSummary]:
    """Rank spectrum-level candidates using unique line coverage and energy agreement.

    Peaks marked as unresolved doublets only contribute half of their line
    intensity, and the affected lines are counted in ``ambiguous_lines`` so that a
    nuclide supported entirely by blended lines cannot look like a confident
    identification.
    """

    summaries: List[NuclideMatchSummary] = []
    for isotope, lines in validate_nuclide_library(library).items():
        total_intensity = sum(float(line["intensity"]) for line in lines)
        matched_intensity = 0.0
        weighted_quality = 0.0
        matched_lines = 0
        ambiguous_lines = 0
        used_peak_indexes: set[int] = set()
        for line in sorted(lines, key=lambda item: float(item["intensity"]), reverse=True):
            reference_energy = float(line["energy"])
            candidates = [
                (abs(peak.energy - reference_energy), index)
                for index, peak in enumerate(peaks)
                if index not in used_peak_indexes and peak.energy > 0
            ]
            if not candidates:
                continue
            difference, peak_index = min(candidates)
            peak_width = float(getattr(peaks[peak_index], "fwhm_kev", 0.0) or 0.0)
            tolerance = max(
                float(absolute_tolerance_kev),
                reference_energy * float(relative_tolerance),
                float(width_tolerance_factor) * peak_width,
            )
            if difference > tolerance:
                continue
            intensity = float(line["intensity"])
            ambiguous = bool(getattr(peaks[peak_index], "interference", None))
            credit = 0.5 if ambiguous else 1.0
            used_peak_indexes.add(peak_index)
            matched_lines += 1
            if ambiguous:
                ambiguous_lines += 1
            matched_intensity += intensity * credit
            weighted_quality += intensity * credit * max(0.0, 1.0 - difference / tolerance)
        if not matched_lines or total_intensity <= 0:
            continue
        coverage = min(matched_intensity / total_intensity, 1.0)
        quality = weighted_quality / matched_intensity if matched_intensity > 0 else 0.0
        score = coverage * (0.7 + 0.3 * quality)
        summaries.append(
            NuclideMatchSummary(
                isotope=isotope,
                matched_lines=matched_lines,
                total_lines=len(lines),
                intensity_coverage=coverage,
                match_score=min(max(score, 0.0), 1.0),
                ambiguous_lines=ambiguous_lines,
            )
        )
    return sorted(
        summaries,
        key=lambda item: (item.match_score, item.matched_lines, item.intensity_coverage),
        reverse=True,
    )


def analyze_spectrum(
    record: SpectrumRecord,
    calibration: Optional[CalibrationModel] = None,
    library: Optional[Dict[str, List[Dict[str, float]]]] = None,
    config: Optional[AnalysisConfig] = None,
    background: Optional[SpectrumRecord] = None,
) -> AnalysisResult:
    """Run the reproducible background -> baseline -> peak -> nuclide pipeline.

    When ``background`` is given, the measured background is subtracted first and
    the SNIP continuum estimate then runs on the residual, so environmental lines
    and the sample's own Compton continuum are handled in that order.
    """

    record = _validated_record(record)
    settings = (config or AnalysisConfig()).validated()
    reference_library = validate_nuclide_library(library or NNDC_LIBRARY)
    background_info: Optional[BackgroundInfo] = None
    analysis_counts = record.counts
    if background is not None:
        record, background_info = subtract_background(
            record,
            background,
            scale_mode=settings.background_scale,
        )
        analysis_counts = record.counts
        if settings.snip_after_background:
            baseline = snip_baseline(analysis_counts, iterations=settings.snip_iterations)
            background_info.continuum = "snip"
        else:
            baseline = np.zeros_like(analysis_counts)
            background_info.continuum = "none"
    else:
        baseline = snip_baseline(analysis_counts, iterations=settings.snip_iterations)
    net_counts = np.clip(analysis_counts - baseline, 0, None)
    candidates = detect_peaks_from_cwt(
        net_counts,
        widths=settings.wavelet_widths,
        min_distance=settings.min_peak_distance,
        threshold_sigma=settings.detection_threshold_sigma,
    )
    peaks = merge_close_peaks(
        refine_peaks(
            record.channels,
            net_counts,
            analysis_counts,
            candidates,
            min_snr=settings.min_snr,
        )
    )
    model = calibration or CalibrationModel()
    match_nuclides(
        peaks,
        model,
        flatten_library(reference_library),
        absolute_tolerance_kev=settings.absolute_tolerance_kev,
        relative_tolerance=settings.relative_tolerance,
        max_candidates=settings.max_candidates,
        width_tolerance_factor=settings.width_tolerance_factor,
        interference_width_factor=settings.interference_width_factor,
    )
    summary = summarize_nuclide_matches(
        peaks,
        reference_library,
        absolute_tolerance_kev=settings.absolute_tolerance_kev,
        relative_tolerance=settings.relative_tolerance,
        width_tolerance_factor=settings.width_tolerance_factor,
    )
    decompose_interference(peaks, record.channels, net_counts, model)
    return AnalysisResult(
        baseline=baseline,
        net_counts=net_counts,
        peaks=peaks,
        nuclide_summary=summary,
        background=background_info,
        analysis_counts=analysis_counts,
    )


def ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
