"""Document loading and PDF rasterization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DocumentPage:
    image: Image.Image
    page_index: int
    source_path: str


def load_document_pages(document_path: str | Path, dpi: int = 200) -> list[DocumentPage]:
    path = Path(document_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(path, dpi=dpi)
    return [DocumentPage(image=_load_image(path), page_index=0, source_path=str(path))]


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_pdf_pages(path: Path, dpi: int = 200) -> list[DocumentPage]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires PyMuPDF (`pymupdf`) to be installed."
        ) from exc

    document = fitz.open(path)
    try:
        pages: list[DocumentPage] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pages.append(DocumentPage(image=image, page_index=page_index, source_path=str(path)))
        return pages
    finally:
        document.close()
