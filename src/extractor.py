"""OCR-first field extraction using word boxes, line grouping, and candidate ranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.ocr import OCRResult, extract_ocr, preprocess_image as preprocess_ocr_image
from src.types import Box, ExtractionResult, FieldPrediction, OCRLine, OCRWord


DATE_REGEXES = (
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%m/%d/%y",
    "%m-%d-%y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%b %d, %Y",
    "%b %d %Y",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %y",
    "%b %d %y",
    "%B %d, %y",
    "%B %d %y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %y",
    "%d %B %y",
)

POSITIVE_TOTAL_TERMS = (
    "grand total",
    "total due",
    "amount due",
    "net total",
    "net amount",
    "balance due",
    "pay",
    "payment",
    "total",
)
NEGATIVE_TOTAL_TERMS = (
    "subtotal",
    "sub total",
    "tax",
    "vat",
    "gst",
    "sst",
    "service",
    "tip",
    "cash",
    "change",
    "card",
    "visa",
    "master",
    "rounding",
    "discount",
    "tendered",
    "quantity",
    "qty",
    "item",
    "no.",
    "number",
    "tel",
    "phone",
    "fax",
    "date",
)
VENDOR_STOP_TERMS = {
    "receipt",
    "invoice",
    "tax invoice",
    "cashier",
    "tel",
    "phone",
    "date",
    "time",
    "total",
    "subtotal",
    "change",
    "cash",
}
DATE_CONTEXT_TERMS = (
    "date",
    "issued",
    "invoice",
    "txn",
    "transaction",
    "receipt date",
    "printed",
    "time",
)
AMOUNT_REGEX = re.compile(
    r"(?:rm|usd|eur|aed|\$)?\s*[-+]?(?:\d[\d., ]*\d|\d*[.,]\d{1,2})",
    re.IGNORECASE,
)
TIME_REGEX = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
PHONE_LIKE_REGEX = re.compile(r"^\+?\d[\d\s().-]{6,}\d$")
INVOICE_ID_REGEX = re.compile(r"(?<!\d)\d{7,}(?!\d)")
VENDOR_PRIOR_COUNTS = {}
for _prior_path in (
    Path(__file__).resolve().parents[1] / "dummy_data" / "train" / "train.jsonl",
    Path(__file__).resolve().parents[1] / "data" / "synthetic_train.jsonl",
):
    if not _prior_path.exists():
        continue
    try:
        with _prior_path.open() as handle:
            for raw_line in handle:
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                vendor = (
                    payload.get("vendor")
                    or payload.get("fields", {}).get("vendor")
                    or ""
                ).strip()
                if vendor:
                    normalized_vendor = re.sub(r"\s+", " ", vendor.lower())
                    VENDOR_PRIOR_COUNTS[normalized_vendor] = (
                        VENDOR_PRIOR_COUNTS.get(normalized_vendor, 0) + 1
                    )
    except OSError:
        continue


@dataclass(frozen=True)
class _Candidate:
    value: str
    score: float
    box: Box | None
    raw_value: str | None = None
    page_index: int = 0


def preprocess_image(image):
    return preprocess_ocr_image(image)


def extract_text(image_path: str) -> str:
    return extract_ocr(image_path).text


def extract_vendor(text: str, words: Iterable[OCRWord] | None = None) -> str | None:
    lines = _lines_from_text_or_words(text, words)
    return _predict_vendor(lines).value or None


def extract_date(text: str) -> str | None:
    lines = _text_to_lines(text)
    return _predict_date(lines).value or None


def normalize_date(date_str: str) -> str:
    normalized, _ = _normalize_date_candidate(date_str)
    return normalized or date_str.strip()


def extract_total(text: str) -> str | None:
    lines = _text_to_lines(text)
    return _predict_total(lines).value or None


def normalize_amount(amount_str: str) -> str:
    normalized = _normalize_amount_token(amount_str)
    return normalized if normalized is not None else amount_str.strip()


def parse_amount(amount_str: str | None) -> float | None:
    normalized = _normalize_amount_token(amount_str) if amount_str else None
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def extract_fields(image_path: str) -> dict[str, str | None]:
    ocr_result = extract_ocr(image_path)
    extraction = extract_fields_from_ocr(ocr_result)
    return {
        "vendor": extraction.vendor.value,
        "date": extraction.date.value,
        "total": extraction.total.value,
        "_ocr_text": ocr_result.text,
    }


def extract_fields_from_ocr(ocr_result: OCRResult, stats: dict[str, object] | None = None) -> ExtractionResult:
    lines = ocr_result.lines or _lines_from_text_or_words(ocr_result.text, ocr_result.words)
    return ExtractionResult(
        vendor=_candidate_to_field("vendor", _predict_vendor(lines, stats=stats)),
        date=_candidate_to_field("date", _predict_date(lines, stats=stats)),
        total=_candidate_to_field("total", _predict_total(lines, stats=stats)),
    )


def _text_to_lines(text: str) -> tuple[OCRLine, ...]:
    lines = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(
            OCRLine(
                text=stripped,
                words=(),
                box=Box(0, index * 10, max(len(stripped) * 8, 1), index * 10 + 8, 0),
                confidence=0.0,
                page_index=0,
            )
        )
    return tuple(lines)


def _lines_from_text_or_words(text: str, words: Iterable[OCRWord] | None) -> tuple[OCRLine, ...]:
    if words:
        grouped: dict[tuple[int, int | None, int | None, int | None], list[OCRWord]] = {}
        for word in words:
            grouped.setdefault(
                (word.page_index, word.block_num, word.paragraph_num, word.line_num),
                [],
            ).append(word)
        lines = []
        for line_words in grouped.values():
            ordered = tuple(sorted(line_words, key=lambda word: (word.left, word.top)))
            lines.append(
                OCRLine(
                    text=" ".join(word.text for word in ordered).strip(),
                    words=ordered,
                    box=Box(
                        left=min(word.left for word in ordered),
                        top=min(word.top for word in ordered),
                        right=max(word.right for word in ordered),
                        bottom=max(word.bottom for word in ordered),
                        page_index=ordered[0].page_index,
                    ),
                    confidence=_mean_confidence(ordered),
                    page_index=ordered[0].page_index,
                )
            )
        return tuple(sorted(lines, key=lambda line: (line.page_index, line.box.top, line.box.left)))
    return _text_to_lines(text)


def _predict_vendor(lines: tuple[OCRLine, ...], stats: dict[str, object] | None = None) -> _Candidate:
    best = _Candidate(value="", score=-1.0, box=None)
    if not lines:
        return best
    known_vendor_set = set()
    if stats is not None:
        raw_known_vendors = stats.get("known_vendors")
        if isinstance(raw_known_vendors, list):
            known_vendor_set = {
                vendor.strip().lower()
                for vendor in raw_known_vendors
                if isinstance(vendor, str) and vendor.strip()
            }

    max_top = max((line.box.bottom for line in lines), default=1)
    min_left = min((line.box.left for line in lines), default=0)
    max_right = max((line.box.right for line in lines), default=1)
    page_width = max(max_right - min_left, 1)
    page_center = min_left + page_width / 2.0
    for index, line in enumerate(lines[:8]):
        text = line.text.strip()
        if not text:
            continue
        lower = text.lower()
        compact_lower = re.sub(r"\s+", " ", lower)
        score = 0.0
        if line.box.top <= max_top * 0.35:
            score += 2.0
        score += max(0.0, 1.2 - 0.18 * index)
        if index < 3:
            score += 0.5
        if len(text) >= 4:
            score += 0.5
        if len(text.split()) <= 6:
            score += 0.4
        alpha_count = sum(char.isalpha() for char in text)
        digit_count = sum(char.isdigit() for char in text)
        alpha_ratio = alpha_count / max(len([char for char in text if not char.isspace()]), 1)
        digit_ratio = digit_count / max(len([char for char in text if not char.isspace()]), 1)
        if alpha_ratio >= 0.65:
            score += 0.5
        elif alpha_ratio < 0.45:
            score -= 0.8
        if digit_ratio > 0.30:
            score -= 1.4
        elif digit_count:
            score -= 0.6
        if any(term in lower for term in VENDOR_STOP_TERMS):
            score -= 1.5
        if re.fullmatch(r"[\W\d_]+", text):
            score -= 1.0
        if line.confidence >= 75:
            score += 0.4
        line_center = (line.box.left + line.box.right) / 2.0
        centeredness = 1.0 - min(abs(line_center - page_center) / (page_width / 2.0), 1.0)
        score += centeredness * 0.8
        if compact_lower in VENDOR_PRIOR_COUNTS:
            score += min(1.6, 0.5 + 0.08 * VENDOR_PRIOR_COUNTS[compact_lower])
        if known_vendor_set:
            candidate_tokens = tuple(token for token in re.split(r"\s+", compact_lower) if token)
            if candidate_tokens and " ".join(candidate_tokens) in known_vendor_set:
                score += 1.0
        if score > best.score:
            best = _Candidate(text, score, line.box, text, line.page_index)
    if best.score < 1.0:
        return _Candidate("", 0.0, None)
    return best


def _predict_date(lines: tuple[OCRLine, ...], stats: dict[str, object] | None = None) -> _Candidate:
    best = _Candidate(value="", score=-1.0, box=None)
    date_format_frequencies = {}
    if stats is not None:
        raw_date_format_frequencies = stats.get("date_format_frequencies")
        if isinstance(raw_date_format_frequencies, dict):
            date_format_frequencies = raw_date_format_frequencies
    for line in lines:
        lower = line.text.lower()
        normalized_lower = re.sub(r"\s+", " ", lower)
        context_bonus = 0.0
        if any(token in normalized_lower for token in DATE_CONTEXT_TERMS):
            context_bonus += 0.6
        if any(token in normalized_lower for token in ("receipt date", "transaction", "txn")):
            context_bonus += 0.2
        has_timestamp = bool(TIME_REGEX.search(line.text))
        for pattern in DATE_REGEXES:
            for match in pattern.finditer(line.text):
                normalized, ambiguous = _normalize_date_candidate(match.group(0))
                if normalized is None:
                    continue
                score = 1.2 + context_bonus + (0.4 if not ambiguous else -0.2)
                matched_pattern = _detect_date_format_pattern(match.group(0))
                if matched_pattern and matched_pattern in date_format_frequencies:
                    score += 0.3
                if has_timestamp:
                    score -= 0.7
                if line.confidence >= 70:
                    score += 0.3
                if score > best.score:
                    best = _Candidate(normalized, score, line.box, match.group(0), line.page_index)
    if best.score < 0.8:
        return _Candidate("", 0.0, None)
    return best


def _predict_total(lines: tuple[OCRLine, ...], stats: dict[str, object] | None = None) -> _Candidate:
    best = _Candidate(value="", score=-1.0, box=None)
    if not lines:
        return best
    amount_q1 = _get_stat_float(stats, "amount_q1")
    amount_q3 = _get_stat_float(stats, "amount_q3")
    amount_mean = _get_stat_float(stats, "amount_mean")
    amount_std = _get_stat_float(stats, "amount_std")
    max_bottom = max((line.box.bottom for line in lines), default=1)
    subtotal_candidates = []
    tax_candidates = []
    for index, line in enumerate(lines):
        lower = re.sub(r"\s+", " ", line.text.lower()).strip()
        amount_tokens = [match.group(0) for match in AMOUNT_REGEX.finditer(line.text)]
        if not amount_tokens:
            continue

        line_score = 0.0
        for term in POSITIVE_TOTAL_TERMS:
            if term in lower:
                line_score += 2.0 if term != "total" else 1.2
        for term in NEGATIVE_TOTAL_TERMS:
            if term in lower:
                line_score -= 1.4
        line_midpoint = (line.box.top + line.box.bottom) / 2.0
        vertical_ratio = line_midpoint / max(max_bottom, 1)
        if vertical_ratio >= 0.60:
            line_score += 0.9
        elif vertical_ratio >= 0.50:
            line_score += 0.4
        if index >= max(0, len(lines) - 4):
            line_score += 0.4

        normalized_amounts = []
        for token in amount_tokens:
            normalized = _normalize_amount_token(token)
            if normalized is None:
                continue
            normalized_amounts.append((token, normalized, float(normalized)))
        if any(term in lower for term in ("subtotal", "sub total")):
            subtotal_candidates.extend(amount for _, _, amount in normalized_amounts)
        if any(term in lower for term in ("tax", "vat", "gst", "sst", "service")):
            tax_candidates.extend(amount for _, _, amount in normalized_amounts)

        for token, normalized, amount in normalized_amounts:
            score = line_score
            if amount > 0:
                score += 0.8
            if abs(amount - round(amount)) > 1e-6:
                score += 0.2
            if amount_q1 is not None and amount_q3 is not None and amount_q1 <= amount <= amount_q3:
                score += 0.5
            if amount_mean is not None and amount_std is not None and amount_std > 0 and abs(amount - amount_mean) > (3 * amount_std):
                score -= 0.3
            if line.confidence >= 65:
                score += 0.2
            if "total" in lower and any(term in lower for term in ("subtotal", "sub total")):
                score -= 0.8
            if _looks_like_date_amount(token):
                score -= 2.2
            if _looks_like_phone_amount(token):
                score -= 2.0
            if _looks_like_invoice_id(token):
                score -= 1.8
            if lower.startswith(("qty", "quantity", "item", "no.", "number")):
                score -= 1.2
            if score > best.score:
                best = _Candidate(normalized, score, line.box, token, line.page_index)
    if subtotal_candidates:
        for subtotal in subtotal_candidates:
            for tax in tax_candidates or (0.0,):
                expected_total = subtotal + tax
                best_amount = parse_amount(best.value)
                if best_amount is not None and abs(best_amount - expected_total) <= 0.06:
                    best = _Candidate(
                        best.value,
                        best.score + 1.2,
                        best.box,
                        best.raw_value,
                        best.page_index,
                    )
                    break
    if best.score < 0.9:
        return _Candidate("", 0.0, None)
    return best


def _candidate_to_field(name: str, candidate: _Candidate) -> FieldPrediction:
    value = candidate.value or None
    confidence = min(max(candidate.score / 3.0, 0.0), 1.0) if value else 0.0
    return FieldPrediction(
        name=name,
        value=value,
        confidence=confidence,
        box=candidate.box,
        page_index=candidate.page_index,
        raw_value=candidate.raw_value,
    )


def _normalize_date_candidate(value: str) -> tuple[str | None, bool]:
    cleaned = re.sub(r"\s+", " ", value.strip().rstrip(".")).replace("Sept", "Sep")
    ambiguous = False
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", cleaned):
        first, second, _ = re.split(r"[-/]", cleaned)
        first_num = int(first)
        second_num = int(second)
        ambiguous = first_num <= 12 and second_num <= 12
        if ambiguous:
            separator = "/" if "/" in cleaned else "-"
            day_first_formats = (
                f"%d{separator}%m{separator}%Y",
                f"%d{separator}%m{separator}%y",
            )
            month_first_formats = (
                f"%m{separator}%d{separator}%Y",
                f"%m{separator}%d{separator}%y",
            )
            for fmt in day_first_formats + month_first_formats:
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                except ValueError:
                    continue
                if parsed.year < 100:
                    year = parsed.year + 2000 if parsed.year < 70 else parsed.year + 1900
                    parsed = parsed.replace(year=year)
                return parsed.strftime("%Y-%m-%d"), ambiguous

    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        if parsed.year < 100:
            year = parsed.year + 2000 if parsed.year < 70 else parsed.year + 1900
            parsed = parsed.replace(year=year)
        return parsed.strftime("%Y-%m-%d"), ambiguous
    return None, ambiguous


def _normalize_amount_token(token: str | None) -> str | None:
    if token is None:
        return None
    cleaned = re.sub(r"(?i)\b(?:rm|usd|eur|aed)\b", "", token)
    cleaned = cleaned.replace(" ", "").replace("\u00a0", "")
    cleaned = re.sub(r"[^\d,.\-+]", "", cleaned)
    if not cleaned or cleaned in {".", ",", "-", "+"}:
        return None

    sign = ""
    if cleaned[0] in "+-":
        sign = cleaned[0]
        cleaned = cleaned[1:]
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        decimal_sep = "." if cleaned.rfind(".") > cleaned.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        cleaned = cleaned.replace(thousands_sep, "")
        if decimal_sep == ",":
            cleaned = cleaned.replace(",", ".")
    elif cleaned.count(",") > 1 and "." not in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) == 2:
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        else:
            cleaned = "".join(parts)
    elif cleaned.count(".") > 1 and "," not in cleaned:
        parts = cleaned.split(".")
        if len(parts[-1]) == 2:
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        else:
            cleaned = "".join(parts)
    elif "," in cleaned and "." not in cleaned:
        integer, decimal = cleaned.rsplit(",", 1)
        cleaned = integer.replace(",", "") + ("." + decimal if len(decimal) in {1, 2} else decimal)
    elif "." in cleaned and "," not in cleaned and cleaned.count(".") == 1:
        integer, decimal = cleaned.rsplit(".", 1)
        if len(decimal) not in {1, 2}:
            cleaned = integer + decimal

    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    try:
        return f"{float(sign + cleaned):.2f}"
    except ValueError:
        return None


def _mean_confidence(words: Iterable[OCRWord]) -> float:
    confidences = [word.confidence for word in words if word.confidence is not None and word.confidence >= 0]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _looks_like_date_amount(token: str) -> bool:
    compact = token.strip()
    return bool(re.search(r"\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?", compact))


def _looks_like_phone_amount(token: str) -> bool:
    compact = token.strip()
    digits = re.sub(r"\D", "", compact)
    return "." not in compact and "," not in compact and len(digits) >= 7 and bool(PHONE_LIKE_REGEX.match(compact))


def _looks_like_invoice_id(token: str) -> bool:
    compact = token.strip()
    digits = re.sub(r"\D", "", compact)
    return "." not in compact and "," not in compact and len(digits) >= 7 and bool(INVOICE_ID_REGEX.search(compact))


def _detect_date_format_pattern(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value.strip().rstrip(".")).replace("Sept", "Sep")
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return fmt
    return None


def _get_stat_float(stats: dict[str, object] | None, key: str) -> float | None:
    if stats is None:
        return None
    value = stats.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
