from __future__ import annotations

import unittest
from unittest.mock import patch

from src.extractor import extract_date, extract_fields, extract_total, normalize_amount
from src.ocr import OCRResult, OCRWord


class ExtractionTests(unittest.TestCase):
    def test_extract_date_normalizes_common_formats(self) -> None:
        self.assertEqual(extract_date("Date: Jan 24, 2024"), "2024-01-24")
        self.assertEqual(extract_date("Issued 2024/01/24"), "2024-01-24")

    def test_extract_total_prefers_total_keyword_and_normalizes(self) -> None:
        text = "Subtotal 10.00\nTOTAL: RM 1,234.50\nCash 1300.00"
        self.assertEqual(extract_total(text), "1234.50")

    def test_normalize_amount_handles_decimal_comma(self) -> None:
        self.assertEqual(normalize_amount("RM 123,45"), "123.45")

    def test_extract_fields_uses_layout_words_for_vendor(self) -> None:
        ocr_result = OCRResult(
            text="Receipt\nDate: 2024-01-24\nTOTAL 45.60",
            words=(
                OCRWord("Bright", 10, 5, 40, 10, line_num=1, block_num=1, paragraph_num=1),
                OCRWord("Market", 55, 5, 45, 10, line_num=1, block_num=1, paragraph_num=1),
                OCRWord("Receipt", 10, 20, 50, 10, line_num=2, block_num=1, paragraph_num=1),
            ),
            image_size=(120, 80),
        )

        with patch("src.extractor.extract_ocr", return_value=ocr_result):
            result = extract_fields("dummy.png")

        self.assertEqual(result["vendor"], "Bright Market")
        self.assertEqual(result["date"], "2024-01-24")
        self.assertEqual(result["total"], "45.60")
        self.assertEqual(result["_ocr_text"], ocr_result.text)


if __name__ == "__main__":
    unittest.main()
