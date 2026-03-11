from __future__ import annotations

import unittest
from unittest.mock import patch

from src.extractor import extract_date, extract_fields, extract_total, normalize_amount
from src.ocr import OCRResult
from src.types import Box, OCRLine, OCRWord


class ExtractionTests(unittest.TestCase):
    def test_extract_date_normalizes_common_formats(self) -> None:
        self.assertEqual(extract_date("Date: Jan 24, 2024"), "2024-01-24")
        self.assertEqual(extract_date("Issued 2024/01/24"), "2024-01-24")
        self.assertEqual(extract_date("Printed 01/24/24"), "2024-01-24")

    def test_extract_total_prefers_total_keyword_and_normalizes(self) -> None:
        text = "Subtotal 10.00\nTOTAL: RM 1,234.50\nCash 1300.00"
        self.assertEqual(extract_total(text), "1234.50")

    def test_normalize_amount_handles_decimal_comma_and_thousands(self) -> None:
        self.assertEqual(normalize_amount("RM 123,45"), "123.45")
        self.assertEqual(normalize_amount("1.234,56"), "1234.56")

    def test_extract_fields_uses_layout_words_for_vendor(self) -> None:
        words = (
            OCRWord("Bright", 10, 5, 40, 10, confidence=95.0, line_num=1, block_num=1, paragraph_num=1),
            OCRWord("Market", 55, 5, 45, 10, confidence=95.0, line_num=1, block_num=1, paragraph_num=1),
            OCRWord("Date:", 10, 20, 35, 10, confidence=90.0, line_num=2, block_num=1, paragraph_num=1),
            OCRWord("2024-01-24", 50, 20, 70, 10, confidence=90.0, line_num=2, block_num=1, paragraph_num=1),
            OCRWord("TOTAL", 10, 35, 40, 10, confidence=90.0, line_num=3, block_num=1, paragraph_num=1),
            OCRWord("45.60", 60, 35, 40, 10, confidence=90.0, line_num=3, block_num=1, paragraph_num=1),
        )
        ocr_result = OCRResult(
            text="Bright Market\nDate: 2024-01-24\nTOTAL 45.60",
            words=words,
            image_size=(120, 80),
            lines=(
                OCRLine("Bright Market", words[:2], Box(10, 5, 100, 15, 0), 95.0, 0),
                OCRLine("Date: 2024-01-24", words[2:4], Box(10, 20, 120, 30, 0), 90.0, 0),
                OCRLine("TOTAL 45.60", words[4:], Box(10, 35, 100, 45, 0), 90.0, 0),
            ),
        )

        with patch("src.extractor.extract_ocr", return_value=ocr_result):
            result = extract_fields("dummy.png")

        self.assertEqual(result["vendor"], "Bright Market")
        self.assertEqual(result["date"], "2024-01-24")
        self.assertEqual(result["total"], "45.60")
        self.assertEqual(result["_ocr_text"], ocr_result.text)


if __name__ == "__main__":
    unittest.main()
