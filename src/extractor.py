"""Receipt field extraction built on top of the OCR layer."""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image

from src.extractors import extract_vendor as extract_vendor_field
from src.ocr import extract_ocr, preprocess_image as preprocess_ocr_image


# Common date patterns found in receipts
DATE_PATTERNS = [
    # ISO: 2024-01-24
    r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
    # US: 01/24/2024 or 01-24-2024
    r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
    # European: 24.01.2024
    r'(\d{1,2}\.\d{1,2}\.\d{4})',
    # Short year: 01/24/24
    r'(\d{1,2}[-/]\d{1,2}[-/]\d{2}\b)',
    # Written: Jan 24, 2024 / January 24, 2024
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})',
    # Written: 24 Jan 2024
    r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})',
]

# Total amount patterns
TOTAL_PATTERNS = [
    # "TOTAL" followed by amount
    r'\b(?:TOTAL|Grand\s*Total|GRAND\s*TOTAL|Amount\s*Due|AMOUNT\s*DUE|NET\s*TOTAL)\b\s*[:\s]*[\$RM]?\s*(\d+[,.]?\d*\.?\d{0,2})',
    # Amount after "TOTAL" on same or next line
    r'\b(?:TOTAL|Total)\b\s*[\$RM]?\s*(\d{1,}[,.]\d{2,})',
    # Dollar sign followed by amount
    r'\$\s*(\d+[,.]?\d*\.\d{2})',
    # "RM" (Malaysian Ringgit) amount
    r'RM\s*(\d+[,.]?\d*\.?\d{0,2})',
]

def preprocess_image(img: Image.Image) -> Image.Image:
    """Basic preprocessing for better OCR results."""
    return preprocess_ocr_image(img)


def extract_text(image_path: str) -> str:
    """Run OCR on a receipt image."""
    return extract_ocr(image_path).text


def extract_vendor(text: str) -> Optional[str]:
    """Extract vendor name from OCR text (usually first meaningful line)."""
    return extract_vendor_field(text)


def extract_date(text: str) -> Optional[str]:
    """Extract date from OCR text."""
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_date(match.group(1))
    return None


def normalize_date(date_str: str) -> str:
    """Try to normalize date to YYYY-MM-DD format."""
    from datetime import datetime
    
    formats = [
        '%Y-%m-%d', '%Y/%m/%d',
        '%m/%d/%Y', '%m-%d-%Y',
        '%d.%m.%Y',
        '%d/%m/%Y', '%d-%m-%Y',
        '%b %d, %Y', '%b %d %Y',
        '%B %d, %Y', '%B %d %Y',
        '%d %b %Y', '%d %B %Y',
    ]
    
    date_str = date_str.strip().rstrip('.')
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return date_str  # Return as-is if can't normalize


def extract_total(text: str) -> Optional[str]:
    """Extract total amount from OCR text."""
    # Try each pattern
    for pattern in TOTAL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Take the last match (usually the final total)
            amount = matches[-1]
            return normalize_amount(amount)
    
    # Fallback: find the largest number that looks like a price
    prices = re.findall(r'(\d{1,6}[,.]\d{2})', text)
    if prices:
        amounts = []
        for p in prices:
            try:
                val = float(p.replace(',', ''))
                amounts.append((val, p))
            except:
                pass
        if amounts:
            # Return the largest amount (likely the total)
            largest = max(amounts, key=lambda x: x[0])
            return normalize_amount(largest[1])
    
    return None


def normalize_amount(amount_str: str) -> str:
    """Normalize amount to standard decimal format."""
    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[^\d,.]', '', amount_str.strip())
    
    # Handle comma as thousands separator (1,234.56)
    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace(',', '')
    # Handle comma as decimal separator (123,45)
    elif ',' in cleaned and cleaned.count(',') == 1:
        parts = cleaned.split(',')
        if len(parts[1]) == 2:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    
    try:
        return f"{float(cleaned):.2f}"
    except:
        return amount_str


def extract_fields(image_path: str) -> dict:
    """Extract all fields from a receipt image."""
    ocr_result = extract_ocr(image_path)
    text = ocr_result.text

    return {
        'vendor': extract_vendor_field(text, ocr_result.words),
        'date': extract_date(text),
        'total': extract_total(text),
        '_ocr_text': text,  # Keep for debugging
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = extract_fields(sys.argv[1])
        print(f"Vendor: {result['vendor']}")
        print(f"Date:   {result['date']}")
        print(f"Total:  {result['total']}")
        print(f"\nOCR Text:\n{result['_ocr_text']}")
