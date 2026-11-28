"""Turn a standard-source certificate into a small, trustworthy nuclide library.

A certificate for a mixed calibration source only names the *parent* nuclides
(``238U``, ``232Th``, ``226Ra`` ...). Gamma spectrometry almost never sees those
parents directly -- it sees daughters such as ``234Th``, ``228Ac`` or ``214Bi``.
Keeping the whole built-in library while analysing such a source floods the
candidate list with nuclides that cannot possibly be in the样品, so this module
keeps only the lines that the certificate actually justifies.
"""

from __future__ import annotations

import html
import math
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 半衰期（年），用于把证书活度校正到测量日期
HALF_LIVES: Dict[str, float] = {
    "co-60": 5.2714,
    "cs-137": 30.08,
    "cs-134": 2.0652,
    "eu-152": 13.517,
    "am-241": 432.6,
    "bi-207": 32.9,
    "pb-210": 22.23,
    "ra-226": 1600.0,
    "th-232": 1.405e10,
    "u-238": 4.468e9,
    "u-235": 7.04e8,
    "k-40": 1.248e9,
}

# 证书上的母体核素 → 实际用于伽马测量的子体（或自身）
CERTIFICATE_PROXIES: Dict[str, List[str]] = {
    # 238U 自身不放特征 γ 线，靠 234Th 的低能线；天然铀中的 235U 也随 238U 存在
    "u-238": ["th-234", "u-235"],
    "th-232": ["ac-228", "pb-212", "tl-208", "bi-212"],
    # 226Ra 自己的 186.2 keV 与 235U 的 185.7 keV 不可分辨，两条都要留在库里
    "ra-226": ["ra-226", "pb-214", "bi-214"],
    "u-235": ["u-235"],
}

# 这些映射需要额外说明，避免让人以为库里多出了没检定的核素
MAPPING_NOTES: Dict[str, str] = {
    "u-235": "U-235 未单独检定，按天然铀丰度（A(235U)/A(238U)≈4.6%）随 238U 计入",
    "ra-226": "186 keV 为 Ra-226 与 U-235 的叠加峰，软件按双线标注而不强行指派",
}

_NUCLIDE_WITH_DASH = re.compile(r"^\s*(?P<element>[A-Za-z]{1,3})\s*[-_]\s*(?P<mass>\d{1,3})\s*$")
_NUCLIDE_COMPACT = re.compile(r"^\s*(?P<mass>\d{1,3})\s*(?P<element>[A-Za-z]{1,3})\s*$")
_NUCLIDE_ELEMENT_FIRST = re.compile(r"^\s*(?P<element>[A-Za-z]{1,3})\s*(?P<mass>\d{1,3})\s*$")
_LABEL = r"(?:\d{1,3}\s*[-_]?\s*[A-Za-z]{1,3}|[A-Za-z]{1,3}\s*[-_]\s*\d{1,3})"
_ENTRY = re.compile(
    r"^\s*(?:[-•·*]\s*)?(?P<label>" + _LABEL + r")\s*[:：=]?\s*"
    r"(?P<value>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(?P<unit>bq\s*(?:/|·|\*)?\s*(?:kg|g|l)?(?:-?1)?)?",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(r"[±]\s*(\d+(?:\.\d+)?)")
_DATE = re.compile(r"((?:19|20)\d{2})\s*[-年./]\s*(\d{1,2})\s*[-月./]\s*(\d{1,2})")
_DATE_KEYWORD = re.compile(r"检定|定值|标定|证书|有效|生产|制造|赋值")
_MASS = re.compile(r"(?:质量|净重|重量|总重|mass)\D{0,12}?(\d+(?:\.\d+)?)\s*(kg|g)\b")
_PAREN_MASS = re.compile(r"[（(]\s*(\d+(?:\.\d+)?)\s*(kg|g)\s*[）)]", re.IGNORECASE)


@dataclass
class CertificateEntry:
    """One nuclide line on the certificate."""

    label: str
    nuclide: str
    activity_bq_per_kg: float
    uncertainty: Optional[float] = None
    proxies: List[str] = field(default_factory=list)
    decay_factor: float = 1.0
    activity_corrected: Optional[float] = None

    def activity_bq(self, mass_kg: Optional[float]) -> Optional[float]:
        if not mass_kg or mass_kg <= 0:
            return None
        return self.activity_bq_per_kg * mass_kg

    def corrected_bq_per_kg(self) -> float:
        """Activity at the measurement date, after decay correction."""

        if self.activity_corrected is None:
            return self.activity_bq_per_kg * self.decay_factor
        return self.activity_corrected


@dataclass
class Certificate:
    entries: List[CertificateEntry] = field(default_factory=list)
    reference_date: Optional[str] = None
    mass_kg: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def _to_date(value: str) -> Optional[date]:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", str(value).strip())
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def decay_factor(nuclide: str, reference_date: Optional[str], target_date: Optional[str]) -> float:
    """Remaining fraction of an activity after decaying to ``target_date``."""

    half_life = HALF_LIVES.get(nuclide)
    start, end = _to_date(reference_date or ""), _to_date(target_date or "")
    if half_life is None or start is None or end is None:
        return 1.0
    elapsed_years = (end - start).days / 365.25
    if elapsed_years <= 0:
        return 1.0
    return 0.5 ** (elapsed_years / half_life)


def apply_decay_correction(certificate: Certificate, target_date: Optional[str]) -> Certificate:
    """Fill each entry's decay factor and corrected activity in place."""

    if not target_date:
        return certificate
    if certificate.reference_date is None:
        certificate.warnings.append("证书未给出定值日期，未做衰变校正。")
        return certificate
    for entry in certificate.entries:
        factor = decay_factor(entry.nuclide, certificate.reference_date, target_date)
        entry.decay_factor = factor
        entry.activity_corrected = entry.activity_bq_per_kg * factor
    return certificate


@dataclass
class CertificateLibrary:
    library: Dict[str, List[Dict[str, float]]]
    certificate: Certificate
    missing_proxies: List[str] = field(default_factory=list)
    target_date: Optional[str] = None

    @property
    def nuclides(self) -> List[str]:
        return sorted(self.library)

    @property
    def line_count(self) -> int:
        return sum(len(lines) for lines in self.library.values())

    def summary_text(self) -> str:
        lines = [
            f"证书核素 {len(self.certificate.entries)} 项 → 库内 {len(self.library)} 种核素 / {self.line_count} 条参考线"
        ]
        for entry in self.certificate.entries:
            via = "、".join(entry.proxies) if entry.proxies else "未找到可用谱线"
            corrected = entry.corrected_bq_per_kg()
            if self.target_date and abs(entry.decay_factor - 1.0) > 1e-6:
                activity = f"{entry.activity_bq_per_kg:g} → {corrected:.3g} Bq/kg（衰减系数 {entry.decay_factor:.4f}）"
            else:
                activity = f"{entry.activity_bq_per_kg:g} Bq/kg"
            total = corrected * self.certificate.mass_kg if self.certificate.mass_kg else None
            extra = f"，{total:.1f} Bq @ {self.certificate.mass_kg:g} kg" if total else ""
            lines.append(f"  {entry.label:>6s}  {activity} → {via}{extra}")
        if self.certificate.reference_date:
            lines.append(f"  证书参考日期：{self.certificate.reference_date}")
        if self.missing_proxies:
            lines.append(f"  库中缺少：{'、'.join(sorted(set(self.missing_proxies)))}（已跳过）")
        notes = [MAPPING_NOTES[name] for name in self.nuclides if name in MAPPING_NOTES]
        for note in dict.fromkeys(notes):
            lines.append(f"  说明：{note}")
        if self.certificate.warnings:
            lines.extend(f"  注意：{item}" for item in self.certificate.warnings)
        return "\n".join(lines)


def normalize_nuclide(label: str) -> Optional[str]:
    """``238U`` / ``U-238`` / ``u238`` → ``u-238``; anything else → None."""

    text = str(label).strip()
    match = (
        _NUCLIDE_WITH_DASH.match(text)
        or _NUCLIDE_COMPACT.match(text)
        or _NUCLIDE_ELEMENT_FIRST.match(text)
    )
    if not match:
        return None
    element = match.group("element")
    element = element[0].upper() + element[1:].lower()
    return f"{element.lower()}-{int(match.group('mass'))}"


def parse_certificate_text(text: str) -> Certificate:
    """Parse the nuclide table of a certificate written as plain text.

    Accepts lines such as ``238U：158 Bq/kg``, ``232Th 65.6 ± 3`` or
    ``K-40 = 535``, plus an optional reference date and sample mass.
    """

    certificate = Certificate()
    date_candidates: List[Tuple[int, str]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        date_match = _DATE.search(line)
        if date_match and not _ENTRY.search(line[:4]):
            year, month, day = (int(part) for part in date_match.groups())
            # “检定/定值/证书”所在行的日期才是证书基准，测量日期只是参考
            priority = 2 if _DATE_KEYWORD.search(line) else 1
            date_candidates.append((priority, f"{year:04d}-{month:02d}-{day:02d}"))
        if certificate.mass_kg is None:
            mass_match = _MASS.search(line)
            if not mass_match and ("源" in line or "样品" in line or "证书" in line):
                mass_match = _PAREN_MASS.search(line)
            if mass_match:
                value = float(mass_match.group(1))
                certificate.mass_kg = value / 1000.0 if mass_match.group(2).lower() == "g" else value

        entry_match = _ENTRY.search(line)
        if not entry_match:
            continue
        nuclide = normalize_nuclide(entry_match.group("label"))
        if nuclide is None:
            continue
        try:
            activity = float(entry_match.group("value"))
        except (TypeError, ValueError):
            continue
        if activity < 0:
            certificate.warnings.append(f"{entry_match.group('label')} 的活度为负，已忽略。")
            continue
        uncertainty_match = _UNCERTAINTY.search(line)
        certificate.entries.append(
            CertificateEntry(
                label=entry_match.group("label").strip(),
                nuclide=nuclide,
                activity_bq_per_kg=activity,
                uncertainty=float(uncertainty_match.group(1)) if uncertainty_match else None,
            )
        )
    if not certificate.entries:
        raise ValueError("没有从文本中识别出任何核素，请按“238U：158 Bq/kg”这样的格式书写。")
    if date_candidates:
        best_priority = max(priority for priority, _ in date_candidates)
        certificate.reference_date = [value for priority, value in date_candidates if priority == best_priority][-1]
    return certificate


def build_library_from_certificate(
    certificate: Certificate,
    base_library: Dict[str, List[Dict[str, float]]],
    target_date: Optional[str] = None,
) -> CertificateLibrary:
    """Keep only the lines that the certificate can explain."""

    library: Dict[str, List[Dict[str, float]]] = {}
    missing: List[str] = []
    apply_decay_correction(certificate, target_date)
    for entry in certificate.entries:
        proxies = CERTIFICATE_PROXIES.get(entry.nuclide, [entry.nuclide])
        entry.proxies = []
        for proxy in proxies:
            if proxy in base_library:
                library[proxy] = base_library[proxy]
                entry.proxies.append(proxy)
            else:
                missing.append(proxy)
    if not library:
        raise ValueError(
            "证书中的核素在现有库中都找不到可用谱线；请先补充这些核素的参考线。"
        )
    return CertificateLibrary(
        library=library,
        certificate=certificate,
        missing_proxies=missing,
        target_date=target_date,
    )


def certificate_library_from_text(
    text: str,
    base_library: Dict[str, List[Dict[str, float]]],
    target_date: Optional[str] = None,
) -> CertificateLibrary:
    return build_library_from_certificate(parse_certificate_text(text), base_library, target_date)


# ---------------- 文件读取 ----------------


def _read_text_any(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", "ignore")


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError("这不是有效的 Word 文档（缺少 word/document.xml）。")
        xml = archive.read("word/document.xml").decode("utf-8", "ignore")
    pattern = re.compile(r"<w:t[^>]*>(.*?)</w:t>|</w:p>|</w:tr>", re.S)
    parts: List[str] = []
    for match in pattern.finditer(xml):
        if match.group(1) is None:
            parts.append("\n")
        else:
            parts.append(html.unescape(match.group(1)))
    return "".join(parts)


def _read_xlsx_text(path: Path) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows_text: List[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f".//{namespace}si"):
                shared.append("".join(node.text or "" for node in item.findall(f".//{namespace}t")))
        sheet_paths = sorted(name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        if not sheet_paths:
            raise ValueError("Excel 文件中没有工作表。")
        for sheet_path in sheet_paths[:1]:
            root = ET.fromstring(archive.read(sheet_path))
            for row in root.findall(f".//{namespace}row"):
                cells: List[str] = []
                for cell in row.findall(f"{namespace}c"):
                    cell_type = cell.get("t", "")
                    raw = cell.findtext(f"{namespace}v", default="")
                    if cell_type == "s" and raw.isdigit() and int(raw) < len(shared):
                        cells.append(shared[int(raw)])
                    elif cell_type == "inlineStr":
                        cells.append("".join(node.text or "" for node in cell.findall(f".//{namespace}t")))
                    else:
                        cells.append(raw)
                if any(cell.strip() for cell in cells):
                    rows_text.append("\t".join(cells))
    return "\n".join(rows_text)


def _read_pdf_text(path: Path) -> str:
    data = path.read_bytes()
    chunks: List[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        raw = match.group(1)
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            continue
        for text_match in re.finditer(rb"\((?:[^()\\]|\\.)*\)", raw):
            piece = text_match.group(0)[1:-1]
            piece = piece.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
            chunks.append(piece.decode("utf-8", "ignore"))
    return "\n".join(chunks)


def read_certificate_file(path: str) -> str:
    """Read a certificate in txt/csv/docx/xlsx/pdf form and return its text."""

    location = Path(path)
    if not location.is_file():
        raise ValueError(f"找不到证书文件：{path}")
    suffix = location.suffix.lower()
    try:
        if suffix == ".docx":
            return _read_docx_text(location)
        if suffix == ".xlsx":
            return _read_xlsx_text(location)
        if suffix == ".pdf":
            text = _read_pdf_text(location)
            if not re.search(r"\d", text):
                raise ValueError(
                    "这个 PDF 里没有可提取的文字（大概率是扫描件或用了内嵌字体）。"
                    "请改用原始 Word/Excel 证书，或把核素与活度直接粘贴进文本框。"
                )
            return text
        if suffix in {".doc", ".xls"}:
            raise ValueError("不支持旧版 .doc/.xls，请另存为 .docx/.xlsx 后再导入。")
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise ValueError(
                "图片证书需要 OCR，本程序不做离线识别。"
                "同一批文件里的 Word 版证书通常含相同的核素表，请优先导入 .docx。"
            )
        return _read_text_any(location)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"文件已损坏或不是有效的 Office 文档：{exc}") from exc


def certificate_library_from_file(
    path: str,
    base_library: Dict[str, List[Dict[str, float]]],
    target_date: Optional[str] = None,
) -> CertificateLibrary:
    return certificate_library_from_text(read_certificate_file(path), base_library, target_date)
