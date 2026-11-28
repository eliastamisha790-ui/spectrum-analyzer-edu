"""Turn a standard-source certificate into a small, trustworthy nuclide library.

A certificate for a mixed calibration source only names the *parent* nuclides
(``238U``, ``232Th``, ``226Ra`` ...). Gamma spectrometry almost never sees those
parents directly -- it sees daughters such as ``234Th``, ``228Ac`` or ``214Bi``.
Keeping the whole built-in library while analysing such a source floods the
candidate list with nuclides that cannot possibly be in the样品, so this module
keeps only the lines that the certificate actually justifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
_MASS = re.compile(r"(?:质量|净重|重\s*量|mass)\D{0,12}?(\d+(?:\.\d+)?)\s*(kg|g)\b", re.IGNORECASE)
_PAREN_MASS = re.compile(r"[（(]\s*(\d+(?:\.\d+)?)\s*(kg|g)\s*[）)]", re.IGNORECASE)


@dataclass
class CertificateEntry:
    """One nuclide line on the certificate."""

    label: str
    nuclide: str
    activity_bq_per_kg: float
    uncertainty: Optional[float] = None
    proxies: List[str] = field(default_factory=list)

    def activity_bq(self, mass_kg: Optional[float]) -> Optional[float]:
        if not mass_kg or mass_kg <= 0:
            return None
        return self.activity_bq_per_kg * mass_kg


@dataclass
class Certificate:
    entries: List[CertificateEntry] = field(default_factory=list)
    reference_date: Optional[str] = None
    mass_kg: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class CertificateLibrary:
    library: Dict[str, List[Dict[str, float]]]
    certificate: Certificate
    missing_proxies: List[str] = field(default_factory=list)

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
            total = entry.activity_bq(self.certificate.mass_kg)
            extra = f"（{total:.1f} Bq @ {self.certificate.mass_kg:g} kg）" if total else ""
            lines.append(f"  {entry.label:>6s}  {entry.activity_bq_per_kg:g} Bq/kg → {via}{extra}")
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
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if certificate.reference_date is None:
            date_match = _DATE.search(line)
            if date_match and not _ENTRY.search(line[:4]):
                year, month, day = (int(part) for part in date_match.groups())
                certificate.reference_date = f"{year:04d}-{month:02d}-{day:02d}"
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
    return certificate


def build_library_from_certificate(
    certificate: Certificate,
    base_library: Dict[str, List[Dict[str, float]]],
) -> CertificateLibrary:
    """Keep only the lines that the certificate can explain."""

    library: Dict[str, List[Dict[str, float]]] = {}
    missing: List[str] = []
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
    return CertificateLibrary(library=library, certificate=certificate, missing_proxies=missing)


def certificate_library_from_text(
    text: str,
    base_library: Dict[str, List[Dict[str, float]]],
) -> CertificateLibrary:
    return build_library_from_certificate(parse_certificate_text(text), base_library)
