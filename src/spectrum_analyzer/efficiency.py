"""Build a full-energy-peak efficiency curve from a certificate plus a spectrum.

The manual workflow is: pick a standard source, read its certificate activity,
measure a peak area, divide by ``activity x emission probability x live time``,
repeat for every line, then fit ln(eps) against ln(E). This module does that for
every certified nuclide at once and reports which points the fit rejects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .certificate import Certificate, apply_decay_correction
from .core import EfficiencyModel, PeakResult

# 证书没写不确定度时采用的默认相对不确定度（标准物质常见 3~10%）
DEFAULT_CERTIFICATE_UNCERTAINTY = 0.05
# 低于该能量的效率受自吸收与探测器窗影响，不再给定量结果
DEFAULT_MIN_QUANT_ENERGY = 100.0


@dataclass
class EfficiencyPointResult:
    nuclide: str
    energy: float
    channel: float
    area: float
    area_uncertainty: float
    intensity: float
    activity_bq: float
    live_time: float
    efficiency: float
    efficiency_uncertainty: float
    residual: float = 0.0
    accepted: bool = True

    def as_text(self) -> str:
        return (
            f"{self.nuclide:7s} {self.energy:8.2f} keV  ε={self.efficiency:.4e}"
            f" ±{self.efficiency_uncertainty * 100:.1f}%  残差 {self.residual * 100:+.1f}%"
        )


@dataclass
class EfficiencyCalibrationResult:
    model: Optional[EfficiencyModel]
    points: List[EfficiencyPointResult] = field(default_factory=list)
    degree: int = 0
    relative_rms: float = 0.0
    skipped: List[str] = field(default_factory=list)
    note: str = ""

    def summary_text(self) -> str:
        if self.model is None:
            return self.note or "未能建立效率刻度。"
        accepted = [point for point in self.points if point.accepted]
        rejected = [point for point in self.points if not point.accepted]
        lines = [
            f"效率刻度：{self.model.describe()}",
            f"采用 {len(accepted)} / {len(self.points)} 个点，拟合阶数 {self.degree}，相对残差 {self.relative_rms * 100:.1f}%",
        ]
        for point in self.points:
            marker = "" if point.accepted else "  ← 未采用"
            lines.append("  " + point.as_text() + marker)
        if rejected:
            lines.append(
                "  未采用的点常见原因：符合相加（级联 γ 线被压低）、峰重叠、低能自吸收或峰误判；"
                "定量时应优先用同一核素自己的线，曲线只用于插值其他能量。"
            )
        if self.skipped:
            lines.append("  跳过：" + "；".join(self.skipped))
        if self.note:
            lines.append("  注意：" + self.note)
        return "\n".join(lines)


def _proxy_activity_map(certificate: Certificate) -> Dict[str, float]:
    """proxy nuclide -> activity (Bq) of the certificate entry that owns it."""

    mapping: Dict[str, float] = {}
    mass = certificate.mass_kg
    for entry in certificate.entries:
        activity_per_kg = entry.corrected_bq_per_kg()
        activity = activity_per_kg * mass if mass else activity_per_kg
        for proxy in entry.proxies:
            mapping.setdefault(proxy, float(activity))
    return mapping


@dataclass
class LineEfficiency:
    """Efficiency of one line, taken from the standard source itself.

    Using the same nuclide's own line avoids the coincidence-summing bias that
    makes a single smooth eps(E) curve inconsistent between cascade nuclides and
    single-gamma nuclides.
    """

    isotope: str
    parent: str
    energy: float
    intensity: float
    area: float
    activity_bq: float
    efficiency: float
    relative_uncertainty: float
    certificate_uncertainty: float = DEFAULT_CERTIFICATE_UNCERTAINTY

    def as_text(self) -> str:
        return (
            f"{self.isotope:7s} {self.energy:8.2f} keV  ε={self.efficiency:.4e}"
            f" ±{self.relative_uncertainty * 100:.1f}%  ← {self.parent}"
        )


@dataclass
class SampleActivity:
    """Activity of one line in the unknown sample."""

    isotope: str
    parent: str
    energy: float
    intensity: float
    area: float
    efficiency: float
    activity_bq: float
    activity_bq_per_kg: float
    relative_uncertainty: float
    flag: str = ""


@dataclass
class QuantificationResult:
    rows: List["SampleActivity"] = field(default_factory=list)
    flagged: List[str] = field(default_factory=list)
    note: str = ""


def build_line_efficiencies(
    peaks: List[PeakResult],
    certificate: Certificate,
    live_time: Optional[float],
    target_date: Optional[str] = None,
) -> List[LineEfficiency]:
    """One efficiency per certificate line, from the standard spectrum itself."""

    if not live_time or live_time <= 0:
        return []
    if target_date:
        apply_decay_correction(certificate, target_date)
    results: List[LineEfficiency] = []
    for entry in certificate.entries:
        activity = entry.corrected_bq_per_kg() * (certificate.mass_kg or 0.0)
        if activity <= 0:
            continue
        if entry.uncertainty is None:
            certificate_uncertainty = DEFAULT_CERTIFICATE_UNCERTAINTY
        elif 0 < entry.uncertainty < 1:
            certificate_uncertainty = float(entry.uncertainty)
        elif entry.activity_bq_per_kg:
            certificate_uncertainty = abs(float(entry.uncertainty)) / entry.activity_bq_per_kg
        else:
            certificate_uncertainty = DEFAULT_CERTIFICATE_UNCERTAINTY
        for peak in peaks:
            if (peak.primary_isotope or "").lower() not in entry.proxies:
                continue
            if peak.is_ambiguous or peak.area <= 0 or peak.reference_intensity <= 0:
                continue
            efficiency = peak.area / (activity * peak.reference_intensity * float(live_time))
            if not math.isfinite(efficiency) or efficiency <= 0:
                continue
            uncertainty = (
                peak.area_uncertainty / peak.area if peak.area_uncertainty > 0 else 1.0 / math.sqrt(peak.area)
            )
            results.append(
                LineEfficiency(
                    isotope=(peak.primary_isotope or "").lower(),
                    parent=entry.label,
                    energy=float(peak.energy),
                    intensity=float(peak.reference_intensity),
                    area=float(peak.area),
                    activity_bq=float(activity),
                    efficiency=float(efficiency),
                    relative_uncertainty=float(uncertainty),
                    certificate_uncertainty=float(certificate_uncertainty),
                )
            )
    return results


def quantify_sample(
    peaks: List[PeakResult],
    line_efficiencies: List[LineEfficiency],
    live_time: Optional[float],
    mass_kg: float,
    min_energy: float = DEFAULT_MIN_QUANT_ENERGY,
    efficiency_systematic: float = 0.0,
) -> QuantificationResult:
    """Apply each line's own efficiency to the unknown spectrum.

    A_smp = 净面积_smp / (ε(E) · Iγ · t_smp)，Bq/kg = A_smp / m_smp。
    分子只取样品自己的峰面积，不涉及标准源面积或证书活度。

    不确定度合成三项：样品峰面积统计、证书相对不确定度、效率刻度的系统项
    （曲线拟合相对残差，`efficiency_systematic`）。只算面积统计会得到
    1~4% 的“假精度”。
    """

    if not live_time or live_time <= 0 or not mass_kg or mass_kg <= 0:
        return QuantificationResult(rows=[], flagged=[], note="缺少活时间或样品质量。")
    by_isotope: Dict[str, List[LineEfficiency]] = {}
    for item in line_efficiencies:
        by_isotope.setdefault(item.isotope, []).append(item)

    results: List[SampleActivity] = []
    flagged: List[str] = []
    for peak in peaks:
        isotope = (peak.primary_isotope or "").lower()
        if isotope not in by_isotope or peak.area <= 0 or peak.reference_intensity <= 0:
            continue
        flag = ""
        if peak.energy < min_energy:
            flag = f"低于 {min_energy:.0f} keV，效率外推（自吸收与低能本底不可控）"
        candidates = by_isotope[isotope]
        reference = min(candidates, key=lambda item: abs(item.energy - peak.energy))
        offset = abs(reference.energy - peak.energy)
        if offset > 3.0:
            flag = (flag + "；" if flag else "") + f"与标准源同核素谱线相差 {offset:.1f} keV，属外推"
        activity = peak.area / (reference.efficiency * peak.reference_intensity * float(live_time))
        area_uncertainty = (
            peak.area_uncertainty / peak.area if peak.area_uncertainty > 0 else 1.0 / math.sqrt(peak.area)
        )
        relative = math.sqrt(
            area_uncertainty ** 2
            + reference.certificate_uncertainty ** 2
            + float(efficiency_systematic) ** 2
        )
        results.append(
            SampleActivity(
                isotope=isotope,
                parent=reference.parent,
                energy=float(peak.energy),
                intensity=float(peak.reference_intensity),
                area=float(peak.area),
                efficiency=reference.efficiency,
                activity_bq=float(activity),
                activity_bq_per_kg=float(activity / mass_kg),
                relative_uncertainty=float(relative),
                flag=flag,
            )
        )
        if flag:
            flagged.append(f"{peak.energy:.1f} keV（{flag}）")
    return QuantificationResult(rows=results, flagged=flagged)


def summarize_activities(rows: List[SampleActivity], include_flagged: bool = False) -> List[Dict[str, float]]:
    """Average the per-line activities of each isotope, weighting by variance.

    带外推标记的线默认不进入汇总：它们仍在报告里单独列出，但不该被当成“结果”
    参与平均（否则 90.7 keV 这种外推点会把 234Th 从 14.6 拉到 61）。
    """

    grouped: Dict[str, List[SampleActivity]] = {}
    for row in rows:
        if row.flag and not include_flagged:
            continue
        grouped.setdefault(row.isotope, []).append(row)
    summary = []
    for isotope, items in grouped.items():
        weights = [1.0 / max(item.relative_uncertainty, 1e-6) ** 2 for item in items]
        total_weight = sum(weights)
        mean = sum(item.activity_bq_per_kg * weight for item, weight in zip(items, weights)) / total_weight
        spread = max(abs(item.activity_bq_per_kg - mean) / mean for item in items) if mean > 0 else 0.0
        summary.append(
            {
                "isotope": isotope,
                "parent": items[0].parent,
                "activity_bq_per_kg": mean,
                "relative_uncertainty": 1.0 / math.sqrt(total_weight),
                "lines": float(len(items)),
                "spread": spread,
            }
        )
    return sorted(summary, key=lambda item: item["activity_bq_per_kg"], reverse=True)


def build_efficiency_from_certificate(
    peaks: List[PeakResult],
    certificate: Certificate,
    live_time: Optional[float],
    library: Dict[str, List[Dict[str, float]]],
    degree: int = 2,
    exclude_isotopes: Optional[List[str]] = None,
    min_points: int = 3,
    max_relative_residual: float = 0.30,
    target_date: Optional[str] = None,
) -> EfficiencyCalibrationResult:
    """One efficiency point per certified line that has a clean peak.

    Points are trimmed iteratively: anything further than
    ``max_relative_residual`` from the current fit is set aside and the curve is
    refitted. Without coincidence-summing corrections a mixed standard shows real
    10-30% scatter (cascade lines such as Tl-208 583 keV are systematically
    depressed), so the threshold is deliberately generous and every point's
    residual is reported rather than hidden.
    """

    if not live_time or live_time <= 0:
        return EfficiencyCalibrationResult(model=None, note="缺少活时间，无法把峰面积换算成计数率。")
    # 用测量日期再校一次衰变：证书缓存可能来自早期版本（未校正），不能默认它已经算过
    if target_date:
        apply_decay_correction(certificate, target_date)
    activity_map = _proxy_activity_map(certificate)
    if not activity_map:
        return EfficiencyCalibrationResult(model=None, note="证书没有可用的核素活度。")
    excluded = {name.lower() for name in (exclude_isotopes or [])}

    points: List[EfficiencyPointResult] = []
    skipped: List[str] = []
    for peak in peaks:
        isotope = (peak.primary_isotope or "").lower()
        if not isotope or isotope in excluded:
            continue
        if peak.is_ambiguous:
            skipped.append(f"{peak.energy:.1f} keV（双线，归属不唯一）")
            continue
        if peak.area_uncertainty <= 0:
            # 面积为窗口求和（高斯拟合未通过）——这类“虚胖”面积会系统性抬高
            # 低能端效率，直接排除比事后剔除更可靠
            skipped.append(f"{peak.energy:.1f} keV（面积未通过高斯拟合，无不确定度）")
            continue
        activity = activity_map.get(isotope)
        if not activity or activity <= 0:
            continue
        intensity = float(peak.reference_intensity)
        if intensity <= 0 or peak.area <= 0:
            continue
        efficiency = peak.area / (activity * intensity * float(live_time))
        if not math.isfinite(efficiency) or efficiency <= 0:
            continue
        relative_uncertainty = (
            peak.area_uncertainty / peak.area if peak.area_uncertainty > 0 else 0.0
        )
        points.append(
            EfficiencyPointResult(
                nuclide=isotope,
                energy=float(peak.energy),
                channel=float(peak.channel),
                area=float(peak.area),
                area_uncertainty=float(peak.area_uncertainty),
                intensity=intensity,
                activity_bq=float(activity),
                live_time=float(live_time),
                efficiency=float(efficiency),
                efficiency_uncertainty=float(relative_uncertainty),
            )
        )

    if len(points) < int(min_points):
        return EfficiencyCalibrationResult(
            model=None,
            points=points,
            skipped=skipped,
            note=f"只有 {len(points)} 个可用效率点，至少需要 {min_points} 个（检查证书活度与刻度）。",
        )

    model, degree_used = _fit_log_log([point for point in points], degree)
    _apply_model(model, points)

    # 逐个剔除最离谱的点，绝不“换点”：迭代过程中重新纳入已被剔除的点会让拟合
    # 在两个错误解之间来回跳（实测会把真正的离群点留在曲线上、把好点子踢出去）
    cap = max(float(max_relative_residual), 0.05)
    # 先用直线做“离群筛选”：高次多项式能弯过去把一个坏点拟合得严丝合缝，
    # 于是最差的点反而显得最好（实测会把好点一个个踢掉），直线的刚性正好防这个
    selection_degree = min(1, max(1, len(points) - 1))
    selection_model, _ = _fit_log_log(points, selection_degree)
    _apply_model(selection_model, points)
    active = [point for point in points if abs(point.residual) <= cap]
    if len(active) < max(int(min_points), degree + 2):
        active = list(points)
    model, degree_used = _fit_log_log(active, degree)
    _apply_model(model, points)
    for point in points:
        point.accepted = abs(point.residual) <= cap

    relative_rms = _rms(points)
    note = ""
    if len([point for point in points if point.accepted]) < int(min_points):
        note = "有效点不足，结果仅供参考。"
    return EfficiencyCalibrationResult(
        model=model,
        points=points,
        degree=degree_used,
        relative_rms=relative_rms,
        skipped=skipped,
        note=note,
    )


def _apply_model(model: EfficiencyModel, points: List[EfficiencyPointResult]) -> None:
    """Recompute each point's residual against the current fit."""

    for point in points:
        predicted = model.efficiency(point.energy)
        point.residual = (
            math.log(point.efficiency / predicted) if predicted > 0 else 0.0
        )


def _rms(points: List[EfficiencyPointResult]) -> float:
    values = [point.residual for point in points if point.accepted]
    if not values:
        values = [point.residual for point in points]
    if not values:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _fit_log_log(
    points: List[EfficiencyPointResult],
    degree: int,
) -> Tuple[EfficiencyModel, int]:
    """Weighted least squares in ln-ln space.

    Weights come from each peak's area uncertainty, so a low-count line does not
    dominate the curve. Robustness is handled outside by trimming points that miss
    the fit; a Huber reweighting step was tried and rejected because a first pass
    over a badly wrong point can bend the curve before the reweighting starts.
    """

    energies = np.asarray([point.energy for point in points], dtype=float)
    efficiencies = np.asarray([point.efficiency for point in points], dtype=float)
    weights = []
    for point in points:
        sigma = point.efficiency_uncertainty
        weights.append(1.0 / sigma if sigma and sigma > 1e-6 else 1.0)
    weights = np.asarray(weights, dtype=float)

    degree = int(max(1, min(degree, len(points) - 1)))
    log_e = np.log(energies)
    log_eff = np.log(efficiencies)
    coefficients = np.polyfit(log_e, log_eff, degree, w=weights)

    model = EfficiencyModel([float(value) for value in coefficients])
    model.set_energy_bounds(list(energies))
    return model, degree
