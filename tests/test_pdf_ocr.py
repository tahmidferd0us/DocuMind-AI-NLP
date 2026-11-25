import io

import pytest
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.parsers import document_parser
from src.parsers.document_parser import OCRUnavailableError, parse_document


def _pdf_with_pages(*kinds):
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    for kind in kinds:
        if kind == "native":
            pdf.drawString(72, 720, "This is selectable native text from a searchable PDF page.")
        elif kind == "image":
            image = Image.new("RGB", (900, 240), "white")
            ImageDraw.Draw(image).text((30, 80), "Scanned page text recognized for OCR", fill="black")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            pdf.drawImage(ImageReader(image_bytes), 72, 560, width=450, height=120)
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def test_native_text_pdf_does_not_call_ocr(monkeypatch):
    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("native text should not invoke OCR")

    monkeypatch.setattr(document_parser, "_ocr_page_text", unexpected_ocr)
    parsed = parse_document(_pdf_with_pages("native"), "native.pdf")

    assert parsed["page_count"] == 1
    assert "selectable native text" in parsed["pages"][0]["cleaned_text"]
    assert parsed["pages"][0]["ocr_used"] is False


def test_image_only_page_uses_ocr_and_records_language(monkeypatch):
    monkeypatch.setattr(document_parser, "_ocr_page_text", lambda _page, _language: "Recognized scanned words from the image.")
    parsed = parse_document(_pdf_with_pages("image"), "scan.pdf")

    page = parsed["pages"][0]
    assert parsed["page_count"] == 1
    assert page["page_number"] == 1
    assert page["cleaned_text"] == "Recognized scanned words from the image."
    assert page["ocr_used"] is True
    assert page["ocr_language"] == "eng"


def test_mixed_pdf_ocr_is_page_scoped_and_preserves_blank_physical_page(monkeypatch):
    calls = []

    def recognize(page, language):
        calls.append((page.number, language))
        return "OCR text from physical page three."

    monkeypatch.setattr(document_parser, "OCR_LANGUAGE", "eng+fra")
    monkeypatch.setattr(document_parser, "_ocr_page_text", recognize)
    parsed = parse_document(_pdf_with_pages("native", "blank", "image"), "mixed.pdf")

    assert parsed["page_count"] == 3
    assert [page["page_number"] for page in parsed["pages"]] == [1, 2, 3]
    assert parsed["pages"][1]["cleaned_text"] == ""
    assert parsed["pages"][0]["ocr_used"] is False
    assert parsed["pages"][2]["ocr_used"] is True
    assert parsed["pages"][2]["ocr_language"] == "eng+fra"
    assert calls == [(2, "eng+fra")]


def test_missing_tesseract_returns_safe_recovery_message(monkeypatch):
    import pytesseract

    class FakePixmap:
        @staticmethod
        def tobytes(_format):
            image = Image.new("RGB", (10, 10), "white")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

    class FakePage:
        @staticmethod
        def get_pixmap(**_kwargs):
            return FakePixmap()

    def missing_engine(*_args, **_kwargs):
        raise pytesseract.pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", missing_engine)
    with pytest.raises(OCRUnavailableError, match="Install Tesseract and its language data"):
        document_parser._ocr_page_text(FakePage(), "eng")


def test_missing_language_data_returns_safe_recovery_message(monkeypatch):
    import pytesseract

    class FakePixmap:
        @staticmethod
        def tobytes(_format):
            image = Image.new("RGB", (10, 10), "white")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

    class FakePage:
        @staticmethod
        def get_pixmap(**_kwargs):
            return FakePixmap()

    def missing_language(*_args, **_kwargs):
        raise pytesseract.pytesseract.TesseractError(1, "Error opening data file at /private/path/eng.traineddata")

    monkeypatch.setattr(pytesseract, "image_to_string", missing_language)
    with pytest.raises(OCRUnavailableError, match="configured OCR language data is unavailable") as error:
        document_parser._ocr_page_text(FakePage(), "eng")
    assert "/private/path" not in str(error.value)
