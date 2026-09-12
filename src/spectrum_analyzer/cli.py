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
    load_nuclide_library,
    load_spectrum_file,
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
    parser.add_argument("spectrum", help="SPE、XLSX、CSV 或 TXT 谱文件")
    parser.add_argument("--slope", type=float, default=1.0, help="线性能量刻度斜率，keV/channel")
    parser.add_argument("--offset", type=float, default=0.0, help="线性能量刻度截距，keV")
    parser.add_argument("--library", help="自定义核素参考库（JSON 或 CSV）")
    parser.add_argument("--snip-iterations", type=int, help="SNIP 本底迭代次数")
    parser.add_argument("--widths", type=parse_widths, default=(2, 4, 6, 8, 10, 12), help="小波宽度，如 2,4,6,8")
    parser.add_argument("--threshold-sigma", type=float, default=1.0, help="寻峰阈值系数")
    parser.add_argument("--min-peak-distance", type=int, default=6, help="最小峰间距（通道）")
    parser.add_argument("--min-snr", type=float, default=1.25, help="最小信噪比")
    parser.add_argument("--absolute-tolerance", type=float, default=0.5, help="核素匹配绝对容差（keV）")
    parser.add_argument("--relative-tolerance", type=float, default=0.002, help="核素匹配相对容差")
    parser.add_argument("--max-candidates", type=int, default=3, help="每个峰最多保留的核素候选数")
    parser.add_argument("--json", dest="json_path", help="将结果写入 JSON 文件")
    return parser


def analyze_file(
    path: str,
    slope: float,
    offset: float,
    library_path: str | None = None,
    config: AnalysisConfig | None = None,
) -> dict:
    record = load_spectrum_file(path)
    library = load_nuclide_library(library_path) if library_path else NNDC_LIBRARY
    settings = (config or AnalysisConfig()).validated()
    result = analyze_spectrum(record, CalibrationModel([slope, offset]), library=library, config=settings)
    return {
        "source": Path(path).name,
        "channels": len(record.channels),
        "real_time_s": record.real_time,
        "live_time_s": record.live_time,
        "calibration": {"slope_kev_per_channel": slope, "offset_kev": offset},
        "reference_library": Path(library_path).name if library_path else "built-in-teaching-library",
        "analysis_config": settings.as_dict(),
        "nuclide_summary": [
            {
                "nuclide": item.isotope,
                "matched_lines": item.matched_lines,
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
                "fwhm_channel": round(peak.fwhm, 4),
                "snr": round(peak.snr, 4),
                "primary_isotope": peak.primary_isotope or None,
                "candidates": peak.nuclide_candidates,
                "match_score": round(peak.confidence, 4),
            }
            for peak in result.peaks
        ],
        "warning": "Teaching/research prototype; results are not certified measurements.",
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = AnalysisConfig(
        snip_iterations=args.snip_iterations,
        wavelet_widths=args.widths,
        detection_threshold_sigma=args.threshold_sigma,
        min_peak_distance=args.min_peak_distance,
        min_snr=args.min_snr,
        absolute_tolerance_kev=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_candidates=args.max_candidates,
    )
    try:
        payload = analyze_file(args.spectrum, args.slope, args.offset, args.library, config)
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
