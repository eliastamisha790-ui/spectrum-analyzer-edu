"""Headless command-line analysis for reproducible demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (
    AnalysisConfig,
    CalibrationModel,
    NNDC_LIBRARY,
    analyze_spectrum,
    auto_calibrate,
    load_nuclide_library,
    load_spectrum_file,
    validate_nuclide_library,
)
from .certificate import certificate_library_from_file
from .efficiency import (
    build_efficiency_from_certificate,
    build_line_efficiencies,
    quantify_sample,
    summarize_activities,
)


def parse_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("小波宽度必须是逗号分隔的整数。") from exc
    if not widths:
        raise argparse.ArgumentTypeError("至少需要一个小波宽度。")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="教学用途的伽马能谱分析命令行工具")
    parser.add_argument("spectrum", nargs="?", help="SPE、XLSX、CSV 或 TXT 谱文件")
    parser.add_argument("--slope", type=float, default=1.0, help="线性能量刻度斜率，keV/channel")
    parser.add_argument("--offset", type=float, default=0.0, help="线性能量刻度截距，keV")
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="枚举候选刻度并按全库命中打分，自动确定能量刻度（忽略 --slope/--offset）",
    )
    parser.add_argument(
        "--allow-unverified-calibration",
        action="store_true",
        help="自动刻度未通过可信度检查时仍然采用（默认改为报错退出）",
    )
    parser.add_argument("--library", help="自定义核素参考库（JSON 或 CSV）")
    parser.add_argument(
        "--certificate",
        help="标准样品源证书（txt/csv/docx/xlsx/pdf），自动收缩为证书涵盖的核素库",
    )
    parser.add_argument(
        "--decay-to",
        help="把证书活度校正到该日期（YYYY-MM-DD）；分析谱时默认取谱的测量日期",
    )
    parser.add_argument("--write-library", help="把最终使用的核素库写成 JSON，便于复用")
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="用证书活度建立效率刻度（需要 --certificate），并把结果写进 JSON",
    )
    parser.add_argument("--write-efficiency", help="把效率刻度（系数与点）写成 JSON")
    parser.add_argument(
        "--standard",
        help="标准源谱文件：给出后用「同一核素自己的线」定效率并定量当前样品",
    )
    parser.add_argument("--sample-mass", type=float, help="样品质量（kg），用于把 Bq 换算成 Bq/kg")
    parser.add_argument("--report", help="把定量结果写成可读文本报告")
    parser.add_argument("--snip-iterations", type=int, help="SNIP 本底迭代次数")
    parser.add_argument("--widths", type=parse_widths, default=(2, 4, 6, 8, 10, 12), help="小波宽度，如 2,4,6,8")
    parser.add_argument("--threshold-sigma", type=float, default=1.0, help="寻峰阈值系数")
    parser.add_argument("--min-peak-distance", type=int, default=6, help="最小峰间距（通道）")
    parser.add_argument("--min-snr", type=float, default=1.25, help="最小信噪比")
    parser.add_argument("--absolute-tolerance", type=float, default=0.5, help="核素匹配绝对容差（keV）")
    parser.add_argument("--relative-tolerance", type=float, default=0.002, help="核素匹配相对容差")
    parser.add_argument("--max-candidates", type=int, default=3, help="每个峰最多保留的核素候选数")
    parser.add_argument("--width-tolerance", type=float, default=0.5, help="容差下限按 FWHM 的倍数放宽")
    parser.add_argument("--interference-width", type=float, default=1.0, help="两条参考线相距小于 FWHM 的该倍数时判为双线")
    parser.add_argument("--background", help="实测本底谱文件（SPE/XLSX/CSV/TXT），先扣除再分析")
    parser.add_argument(
        "--background-scale",
        choices=("live", "real", "none"),
        default="live",
        help="本底归一方式：live=按活时间比，real=按实时间比，none=1:1 直扣",
    )
    parser.add_argument("--no-snip-after-background", action="store_true", help="扣除本底后不再做 SNIP 连续本底估计")
    parser.add_argument("--json", dest="json_path", help="将结果写入 JSON 文件")
    return parser


def _read_text_file(path: str) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ValueError(f"无法读取文件：{exc}") from exc
    raise ValueError("文件编码无法识别，请使用 UTF-8 或 GB18030。")


def convert_certificate(
    certificate_path: str,
    base_library_path: str | None = None,
    output_path: str | None = None,
    target_date: str | None = None,
) -> dict:
    """Turn a certificate into a nuclide library without analysing a spectrum."""

    base = load_nuclide_library(base_library_path) if base_library_path else NNDC_LIBRARY
    result = certificate_library_from_file(certificate_path, base, target_date=target_date)
    payload = {
        "certificate": Path(certificate_path).name,
        "entries": [
            {
                "label": entry.label,
                "nuclide": entry.nuclide,
                "activity_bq_per_kg": entry.activity_bq_per_kg,
                "decay_factor": round(entry.decay_factor, 6),
                "activity_corrected_bq_per_kg": round(entry.corrected_bq_per_kg(), 6),
                "uncertainty": entry.uncertainty,
                "proxies": entry.proxies,
                "activity_bq": entry.activity_bq(result.certificate.mass_kg),
            }
            for entry in result.certificate.entries
        ],
        "reference_date": result.certificate.reference_date,
        "target_date": result.target_date,
        "mass_kg": result.certificate.mass_kg,
        "nuclides": result.nuclides,
        "line_count": result.line_count,
        "missing_proxies": sorted(set(result.missing_proxies)),
        "library": result.library,
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(validate_nuclide_library(result.library), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["written_to"] = str(destination.resolve())
    return payload


def quantify_from_standard(
    standard_path: str,
    sample_path: str,
    certificate_path: str,
    sample_mass_kg: float,
    library_path: str | None = None,
    decay_to: str | None = None,
) -> dict:
    """Per-line efficiency from the standard, applied to the unknown sample."""

    base = load_nuclide_library(library_path) if library_path else NNDC_LIBRARY
    standard = load_spectrum_file(standard_path)
    sample = load_spectrum_file(sample_path)
    certificate_result = certificate_library_from_file(
        certificate_path, base, target_date=decay_to or standard.measurement_date
    )
    library = certificate_result.library

    def analyse(record):
        probe = analyze_spectrum(record, None, library=library)
        auto = auto_calibrate(probe.peaks, library)
        if auto.model is None:
            raise ValueError(f"{record.filepath} 自动刻度失败：{auto.note}")
        return analyze_spectrum(record, auto.model, library=library), auto

    standard_result, standard_auto = analyse(standard)
    lines = build_line_efficiencies(
        standard_result.peaks,
        certificate_result.certificate,
        standard.live_time,
        target_date=decay_to or standard.measurement_date,
    )
    if not lines:
        raise ValueError("标准源谱里没有可用于定效率的检定谱线。")

    sample_result, sample_auto = analyse(sample)
    rows = quantify_sample(sample_result.peaks, lines, sample.live_time, sample_mass_kg)
    summary = summarize_activities(rows)

    return {
        "standard": {
            "file": Path(standard_path).name,
            "live_time_s": standard.live_time,
            "measurement_date": standard.measurement_date,
            "calibration": [float(value) for value in standard_auto.model.coefficients],
            "line_efficiencies": [
                {
                    "nuclide": item.isotope,
                    "parent": item.parent,
                    "energy_kev": round(item.energy, 3),
                    "intensity": item.intensity,
                    "area": round(item.area, 1),
                    "efficiency": float(f"{item.efficiency:.6e}"),
                    "relative_uncertainty": round(item.relative_uncertainty, 4),
                }
                for item in lines
            ],
        },
        "sample": {
            "file": Path(sample_path).name,
            "live_time_s": sample.live_time,
            "measurement_date": sample.measurement_date,
            "mass_kg": sample_mass_kg,
            "calibration": [float(value) for value in sample_auto.model.coefficients],
        },
        "lines": [
            {
                "nuclide": row.isotope,
                "parent": row.parent,
                "energy_kev": round(row.energy, 3),
                "area": round(row.area, 1),
                "efficiency": float(f"{row.efficiency:.6e}"),
                "activity_bq": round(row.activity_bq, 2),
                "activity_bq_per_kg": round(row.activity_bq_per_kg, 2),
                "relative_uncertainty": round(row.relative_uncertainty, 4),
            }
            for row in rows
        ],
        "summary": [
            {
                "nuclide": item["isotope"],
                "parent": item["parent"],
                "activity_bq_per_kg": round(item["activity_bq_per_kg"], 2),
                "relative_uncertainty": round(item["relative_uncertainty"], 4),
                "lines": int(item["lines"]),
                "line_spread": round(item["spread"], 4),
            }
            for item in summary
        ],
        "warning": "教学/科研原型结果，需人工复核；未做符合相加、自吸收与几何差异校正。",
    }


def render_quantification_report(payload: dict) -> str:
    """Human-readable report for the per-line quantification."""

    standard = payload["standard"]
    sample = payload["sample"]
    lines = [
        "沉积物样品核素定量报告（逐核素效率法）",
        "=" * 60,
        f"生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "一、测量与刻度",
        f"  标准源：{standard['file']}  活时间 {standard['live_time_s']:.0f} s  测量日期 {standard['measurement_date']}",
        f"          能量刻度 {standard['calibration'][0]:.6f}·x + {standard['calibration'][1]:.6f}",
        f"  样品  ：{sample['file']}  质量 {sample['mass_kg'] * 1000:.2f} g  活时间 {sample['live_time_s']:.0f} s",
        f"          测量日期 {sample['measurement_date']}",
        f"          能量刻度 {sample['calibration'][0]:.6f}·x + {sample['calibration'][1]:.6f}",
        "",
        "二、标准源逐线效率（由检定活度反算，同一几何）",
        f"  {'核素':<8}{'母体':<7}{'能量 keV':>10}{'发射概率':>10}{'峰面积':>12}{'效率':>12}{'不确定度':>10}",
    ]
    for item in sorted(standard["line_efficiencies"], key=lambda x: x["energy_kev"]):
        lines.append(
            f"  {item['nuclide']:<8}{item['parent']:<7}{item['energy_kev']:>10.2f}{item['intensity']:>10.3f}"
            f"{item['area']:>12.0f}{item['efficiency']:>12.4e}{item['relative_uncertainty'] * 100:>9.1f}%"
        )
    lines += [
        "",
        "三、样品逐线活度",
        f"  {'核素':<8}{'母体':<7}{'能量 keV':>10}{'峰面积':>10}{'活度 Bq':>10}{'Bq/kg':>10}{'不确定度':>10}",
    ]
    for row in sorted(payload["lines"], key=lambda x: x["energy_kev"]):
        lines.append(
            f"  {row['nuclide']:<8}{row['parent']:<7}{row['energy_kev']:>10.2f}{row['area']:>10.0f}"
            f"{row['activity_bq']:>10.2f}{row['activity_bq_per_kg']:>10.2f}{row['relative_uncertainty'] * 100:>9.1f}%"
        )
    lines += [
        "",
        "四、按核素汇总（同核素多线按方差加权）",
        f"  {'核素':<8}{'母体':<7}{'Bq/kg':>10}{'不确定度':>10}{'线数':>6}{'线间离散':>10}",
    ]
    for item in payload["summary"]:
        lines.append(
            f"  {item['nuclide']:<8}{item['parent']:<7}{item['activity_bq_per_kg']:>10.2f}"
            f"{item['relative_uncertainty'] * 100:>9.1f}%{item['lines']:>6}{item['line_spread'] * 100:>9.1f}%"
        )
    summary = {item["nuclide"]: item for item in payload["summary"]}
    lines += ["", "五、自洽性检查"]
    if "pb-214" in summary and "bi-214" in summary:
        ratio = summary["bi-214"]["activity_bq_per_kg"] / summary["pb-214"]["activity_bq_per_kg"]
        lines.append(
            f"  226Ra 系：214Pb {summary['pb-214']['activity_bq_per_kg']:.2f} vs 214Bi "
            f"{summary['bi-214']['activity_bq_per_kg']:.2f} Bq/kg，比值 {ratio:.3f}（平衡应为 1.00）"
        )
    if "th-234" in summary and "pb-214" in summary:
        relation = "基本平衡" if summary["th-234"]["activity_bq_per_kg"] >= summary["pb-214"]["activity_bq_per_kg"] * 0.8 else "不平衡（铀相对镭亏损）"
        lines.append(
            f"  238U 对 226Ra：234Th {summary['th-234']['activity_bq_per_kg']:.2f} vs "
            f"214Pb {summary['pb-214']['activity_bq_per_kg']:.2f} Bq/kg → {relation}"
        )
    if "pb-210" in summary and "pb-214" in summary:
        ratio = summary["pb-210"]["activity_bq_per_kg"] / summary["pb-214"]["activity_bq_per_kg"]
        extra = "（存在 210Pb 过剩，可用于沉积年代指示）" if ratio > 1.2 else ""
        lines.append(
            f"  210Pb 对 226Ra：{summary['pb-210']['activity_bq_per_kg']:.2f} vs "
            f"{summary['pb-214']['activity_bq_per_kg']:.2f} Bq/kg，比值 {ratio:.2f}{extra}"
        )
    lines += [
        "",
        "六、限制声明",
        "  1) 效率来自同一核素的检定谱线，未做符合相加、自吸收与几何差异校正；",
        "  2) 标准源与样品质量接近、同探测器同几何时，系统偏差可部分抵消；",
        "  3) 只有单条线且峰形不佳的核素不作为定量依据（见线间离散）；",
        "  4) 本软件为教学/科研原型，结果需人工复核，不得用于监管或合规决策。",
    ]
    return "\n".join(lines) + "\n"


def analyze_file(
    path: str,
    slope: float,
    offset: float,
    library_path: str | None = None,
    config: AnalysisConfig | None = None,
    background_path: str | None = None,
    auto_calibrate_enabled: bool = False,
    allow_unverified: bool = False,
    certificate_path: str | None = None,
    write_library: str | None = None,
    decay_to: str | None = None,
    efficiency: bool = False,
    write_efficiency: str | None = None,
) -> dict:
    record = load_spectrum_file(path)
    background = load_spectrum_file(background_path) if background_path else None
    base_library = load_nuclide_library(library_path) if library_path else NNDC_LIBRARY
    certificate_payload = None
    certificate_object = None
    library = base_library
    if certificate_path:
        target_date = decay_to or record.measurement_date
        certificate_result = certificate_library_from_file(
            certificate_path, base_library, target_date=target_date
        )
        library = certificate_result.library
        certificate_object = certificate_result.certificate
        certificate_payload = {
            "file": Path(certificate_path).name,
            "reference_date": certificate_result.certificate.reference_date,
            "target_date": certificate_result.target_date,
            "mass_kg": certificate_result.certificate.mass_kg,
            "entries": [
                {
                    "label": entry.label,
                    "nuclide": entry.nuclide,
                    "activity_bq_per_kg": entry.activity_bq_per_kg,
                    "decay_factor": round(entry.decay_factor, 6),
                    "activity_corrected_bq_per_kg": round(entry.corrected_bq_per_kg(), 6),
                    "proxies": entry.proxies,
                }
                for entry in certificate_result.certificate.entries
            ],
            "missing_proxies": sorted(set(certificate_result.missing_proxies)),
        }
    if write_library:
        destination = Path(write_library)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(validate_nuclide_library(library), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    settings = (config or AnalysisConfig()).validated()

    auto_payload = None
    if auto_calibrate_enabled:
        # 寻峰与刻度无关，先用临时刻度拿峰，再让全库投票决定刻度，最后重跑匹配
        probe = analyze_spectrum(
            record,
            CalibrationModel([1.0, 0.0]),
            library=library,
            config=settings,
            background=background,
        )
        auto_result = auto_calibrate(
            probe.peaks,
            library,
            absolute_tolerance_kev=settings.absolute_tolerance_kev,
            relative_tolerance=settings.relative_tolerance,
        )
        if auto_result.model is None:
            raise ValueError(f"自动能量刻度失败：{auto_result.note}")
        if not auto_result.credible and not allow_unverified:
            raise ValueError(
                "自动能量刻度未通过可信度检查："
                f"{auto_result.note or '候选解不唯一'}"
                "（如确认采用，请加 --allow-unverified-calibration）"
            )
        slope = float(auto_result.model.coefficients[0])
        offset = float(auto_result.model.coefficients[1])
        auto_payload = {
            "score": round(auto_result.score, 4),
            "runner_up_score": round(auto_result.runner_up_score, 4),
            "ambiguous": auto_result.ambiguous,
            "credible": auto_result.credible,
            "normalized_rms": round(auto_result.normalized_rms, 4),
            "residual_rms_kev": round(auto_result.residual_rms_kev, 4),
            "hypotheses": auto_result.hypotheses,
            "note": auto_result.note,
            "matched_lines": [
                {
                    "nuclide": item.isotope,
                    "energy_kev": item.energy,
                    "channel": round(item.channel, 3),
                    "residual_kev": round(item.residual_kev, 4),
                }
                for item in auto_result.matched
            ],
        }

    result = analyze_spectrum(
        record,
        CalibrationModel([slope, offset]),
        library=library,
        config=settings,
        background=background,
    )

    efficiency_payload = None
    if (efficiency or write_efficiency) and certificate_object is not None:
        live_time = record.live_time or record.real_time
        efficiency_result = build_efficiency_from_certificate(
            result.peaks,
            certificate_object,
            live_time,
            library,
            target_date=record.measurement_date,
        )
        if efficiency_result.model is not None:
            efficiency_payload = {
                "coefficients": [float(value) for value in efficiency_result.model.coefficients],
                "degree": efficiency_result.degree,
                "relative_rms": round(efficiency_result.relative_rms, 4),
                "accepted_points": len([p for p in efficiency_result.points if p.accepted]),
                "total_points": len(efficiency_result.points),
                "points": [
                    {
                        "nuclide": point.nuclide,
                        "energy_kev": round(point.energy, 3),
                        "area": round(point.area, 2),
                        "activity_bq": round(point.activity_bq, 4),
                        "intensity": point.intensity,
                        "efficiency": float(f"{point.efficiency:.6e}"),
                        "residual": round(point.residual, 4),
                        "accepted": point.accepted,
                    }
                    for point in efficiency_result.points
                ],
                "note": efficiency_result.note,
            }
            if write_efficiency:
                destination = Path(write_efficiency)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(efficiency_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        else:
            efficiency_payload = {"error": efficiency_result.note}
    payload = {
        "source": Path(path).name,
        "channels": len(record.channels),
        "real_time_s": record.real_time,
        "live_time_s": record.live_time,
        "calibration": {
            "slope_kev_per_channel": slope,
            "offset_kev": offset,
            "source": "auto" if auto_payload else "manual",
            "auto": auto_payload,
        },
        "reference_library": (
            f"certificate:{Path(certificate_path).name}"
            if certificate_path
            else (Path(library_path).name if library_path else "built-in-teaching-library")
        ),
        "certificate": certificate_payload,
        "efficiency": efficiency_payload,
        "library_nuclides": sorted(library),
        "analysis_config": settings.as_dict(),
        "nuclide_summary": [
            {
                "nuclide": item.isotope,
                "matched_lines": item.matched_lines,
                "ambiguous_lines": item.ambiguous_lines,
                "total_reference_lines": item.total_lines,
                "intensity_coverage": round(item.intensity_coverage, 4),
                "match_score": round(item.match_score, 4),
            }
            for item in result.nuclide_summary
        ],
        "peaks": [
            {
                "channel": round(peak.channel, 4),
                "energy_kev": round(peak.energy, 4),
                "area": round(peak.area, 4),
                "area_uncertainty": round(peak.area_uncertainty, 4),
                "fwhm_channel": round(peak.fwhm, 4),
                "fwhm_kev": round(peak.fwhm_kev, 4),
                "snr": round(peak.snr, 4),
                "primary_isotope": peak.primary_isotope or None,
                "candidates": peak.nuclide_candidates,
                "match_score": round(peak.confidence, 4),
                "unresolved_doublet": [
                    {
                        "nuclide": line.isotope,
                        "energy_kev": line.energy,
                        "intensity": line.intensity,
                        "delta_kev": round(line.delta_kev, 4),
                        "intensity_share": round(line.weight, 4),
                    }
                    for line in peak.interference
                ]
                if peak.is_ambiguous
                else [],
                "decomposition": {
                    "resolved": peak.decomposition_resolved,
                    "note": peak.decomposition_note,
                    "components": [
                        {
                            "nuclide": component.isotope,
                            "energy_kev": component.energy,
                            "area": round(component.area, 4),
                            "area_uncertainty": round(component.area_uncertainty, 4),
                            "share": round(component.share, 4),
                            "share_uncertainty": round(component.share_uncertainty, 4),
                        }
                        for component in peak.components
                    ],
                }
                if peak.is_ambiguous
                else None,
            }
            for peak in result.peaks
        ],
        "warning": "Teaching/research prototype; results are not certified measurements.",
    }
    if result.background is not None:
        info = result.background
        payload["background"] = {
            "file": Path(info.filepath).name,
            "scale_mode": info.scale_mode,
            "scale": round(info.scale, 6),
            "resampled": info.resampled,
            "clipped_channels": info.clipped_channels,
            "clipped_fraction": round(info.clipped_fraction, 4),
            "continuum_after_subtraction": info.continuum,
            "warnings": list(info.warnings),
        }
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 只做证书转换：不提供谱文件，但要给出 --certificate 与 --write-library
    if not getattr(args, "spectrum", None):
        if not (args.certificate and args.write_library):
            parser.error("需要提供谱文件；只做证书转换时请同时给出 --certificate 与 --write-library")
        try:
            payload = convert_certificate(
                args.certificate, args.library, args.write_library, args.decay_to
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # 用标准源逐线定效率再定量样品（方向 A）
    if args.standard:
        if not args.certificate:
            parser.error("用 --standard 定量时需要同时给出 --certificate")
        if not args.sample_mass or args.sample_mass <= 0:
            parser.error("用 --standard 定量时需要 --sample-mass（kg）")
        try:
            payload = quantify_from_standard(
                args.standard,
                args.spectrum,
                args.certificate,
                args.sample_mass,
                args.library,
                args.decay_to,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.report:
            destination = Path(args.report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(render_quantification_report(payload), encoding="utf-8")
            print(f"报告已写入：{destination.resolve()}")
            print()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    config = AnalysisConfig(
        snip_iterations=args.snip_iterations,
        wavelet_widths=args.widths,
        detection_threshold_sigma=args.threshold_sigma,
        min_peak_distance=args.min_peak_distance,
        min_snr=args.min_snr,
        absolute_tolerance_kev=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_candidates=args.max_candidates,
        width_tolerance_factor=args.width_tolerance,
        interference_width_factor=args.interference_width,
        background_scale=args.background_scale,
        snip_after_background=not args.no_snip_after_background,
    )
    try:
        payload = analyze_file(
            args.spectrum,
            args.slope,
            args.offset,
            args.library,
            config,
            background_path=args.background,
            auto_calibrate_enabled=args.auto_calibrate,
            allow_unverified=args.allow_unverified_calibration,
            certificate_path=args.certificate,
            write_library=args.write_library,
            decay_to=args.decay_to,
            efficiency=args.efficiency,
            write_efficiency=args.write_efficiency,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
        print(destination.resolve())
    else:
        print(output)


if __name__ == "__main__":
    main()
