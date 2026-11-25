import io
import logging
import re
import unicodedata
from typing import Dict, Any, List, Union
from pathlib import Path
from ..config import OCR_DPI, OCR_LANGUAGE, OCR_MIN_TEXT_CHARS, TESSERACT_CMD, TESSDATA_PREFIX

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    pass


class OCRProcessingError(RuntimeError):
    pass

def clean_text(text: str) -> str:
    """Normalises whitespace, fixes encoding artifacts, and removes boilerplate noise."""
    if not text:
        return ""
    
    # Unicode normalization (NFKC decomposes compatibility chars and standardizes)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace non-breaking spaces, zero-width spaces, special whitespace
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    
    # Normalize varied bullet points
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219]', '•', text)
    
    # Normalize quotation marks and apostrophes
    text = re.sub(r'[\u2018\u2019\u201A\u201B]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    
    # Fix hyphenation at line breaks (e.g., "docu-\nment" -> "document")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normalize multiple newlines (keep max 2 newlines for paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize consecutive spaces/tabs within lines
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    return text.strip()

def _read_pdf_bytes(file_source: Union[bytes, io.BytesIO, str, Path]) -> bytes:
    if isinstance(file_source, bytes):
        return file_source
    if isinstance(file_source, (str, Path)):
        return Path(file_source).read_bytes()
    file_source.seek(0)
    return file_source.read()


def _page_text(page_number: int, raw_text: str, has_images: bool = False) -> Dict[str, Any]:
    cleaned = clean_text(raw_text)
    return {
        "page_number": page_number,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "word_count": len(cleaned.split()),
        "char_count": len(cleaned),
        "ocr_used": False,
        "ocr_language": None,
        "_has_images": has_images,
    }


def _extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return [
                _page_text(idx, page.extract_text(layout=True) or page.extract_text() or "", bool(page.images))
                for idx, page in enumerate(pdf.pages, start=1)
            ]
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return [_page_text(idx, page.extract_text() or "") for idx, page in enumerate(reader.pages, start=1)]


def _ocr_page_text(page, language: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise OCRUnavailableError("OCR support is not installed on the analysis service. Ask the administrator to install the documented OCR dependencies.") from error

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    pixmap = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
        config = f'--tessdata-dir "{TESSDATA_PREFIX}"' if TESSDATA_PREFIX else ""
        try:
            return pytesseract.image_to_string(image, lang=language, config=config)
        except pytesseract.pytesseract.TesseractNotFoundError as error:
            raise OCRUnavailableError("OCR is not configured on the analysis service. Install Tesseract and its language data, then retry.") from error
        except pytesseract.pytesseract.TesseractError as error:
            error_text = str(error).lower()
            if "language" in error_text or "tessdata" in error_text or "data file" in error_text:
                raise OCRUnavailableError("The configured OCR language data is unavailable. Ask the administrator to install the requested language data.") from error
            raise OCRProcessingError("OCR could not read a scanned page. Try a clearer PDF or contact the administrator.") from error
        except Exception as error:
            logger.warning("OCR page processing failed", extra={"error_type": type(error).__name__})
            raise OCRProcessingError("OCR could not read a scanned page. Try a clearer PDF or contact the administrator.") from error


def _apply_pdf_ocr(pdf_bytes: bytes, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        idx for idx, page in enumerate(pages)
        if page["_has_images"] and len(page["cleaned_text"]) < OCR_MIN_TEXT_CHARS
    ]
    if not candidates:
        return pages

    try:
        import pymupdf
    except ImportError as error:
        raise OCRUnavailableError("OCR support is not installed on the analysis service. Ask the administrator to install the documented OCR dependencies.") from error

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            for idx in candidates:
                recognized = clean_text(_ocr_page_text(document[idx], OCR_LANGUAGE))
                if recognized:
                    pages[idx].update({
                        "raw_text": recognized,
                        "cleaned_text": recognized,
                        "word_count": len(recognized.split()),
                        "char_count": len(recognized),
                        "ocr_used": True,
                        "ocr_language": OCR_LANGUAGE,
                    })
        finally:
            document.close()
    except (OCRUnavailableError, OCRProcessingError):
        raise
    except Exception as error:
        logger.warning("OCR PDF initialization failed", extra={"error_type": type(error).__name__})
        raise OCRProcessingError("OCR could not read this scanned PDF. Try a clearer PDF or contact the administrator.") from error
    return pages


def parse_pdf(file_source: Union[bytes, io.BytesIO, str, Path]) -> List[Dict[str, Any]]:
    """Extracts native PDF text first and OCRs image pages with sparse text."""
    pdf_bytes = _read_pdf_bytes(file_source)
    pages = _apply_pdf_ocr(pdf_bytes, _extract_pdf_pages(pdf_bytes))
    for page in pages:
        page.pop("_has_images", None)
    return pages

    # Try pdfplumber first
    try:
        import pdfplumber
        stream.seek(0)
        with pdfplumber.open(stream) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_text(layout=True) or page.extract_text() or ""
                cleaned = clean_text(extracted)
                if cleaned:
                    pages.append({
                        "page_number": idx,
                        "raw_text": extracted,
                        "cleaned_text": cleaned,
                        "word_count": len(cleaned.split()),
                        "char_count": len(cleaned)
                    })
    except Exception:
        # Fallback to pypdf
        import pypdf
        stream.seek(0)
        reader = pypdf.PdfReader(stream)
        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = clean_text(extracted)
            if cleaned:
                pages.append({
                    "page_number": idx,
                    "raw_text": extracted,
                    "cleaned_text": cleaned,
                    "word_count": len(cleaned.split()),
                    "char_count": len(cleaned)
                })
                
    return pages

def parse_docx(file_source: Union[bytes, io.BytesIO, str, Path]) -> List[Dict[str, Any]]:
    """Extracts paragraphs and tables from a Word DOCX document."""
    import docx
    
    if isinstance(file_source, bytes):
        stream = io.BytesIO(file_source)
    elif isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            stream = io.BytesIO(f.read())
    else:
        stream = file_source

    stream.seek(0)
    doc = docx.Document(stream)
    
    paragraphs = []
    for p in doc.paragraphs:
        if p.text.strip():
            paragraphs.append(p.text.strip())
            
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    full_raw = "\n\n".join(paragraphs)
    full_cleaned = clean_text(full_raw)
    
    # DOCX does not have fixed static pages; represent as unified document content
    return [{
        "page_number": 1,
        "raw_text": full_raw,
        "cleaned_text": full_cleaned,
        "word_count": len(full_cleaned.split()),
        "char_count": len(full_cleaned)
    }]

def parse_txt(file_source: Union[bytes, io.BytesIO, str, Path]) -> List[Dict[str, Any]]:
    """Extracts plain text with multi-encoding fallback."""
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_source, io.BytesIO):
        raw_bytes = file_source.getvalue()
    else:
        raw_bytes = file_source

    # Detect encoding
    encodings = ["utf-8", "utf-16", "latin-1", "cp1252", "iso-8859-1"]
    decoded = None
    for enc in encodings:
        try:
            decoded = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
            
    if decoded is None:
        decoded = raw_bytes.decode("utf-8", errors="ignore")

    cleaned = clean_text(decoded)
    return [{
        "page_number": 1,
        "raw_text": decoded,
        "cleaned_text": cleaned,
        "word_count": len(cleaned.split()),
        "char_count": len(cleaned)
    }]

def parse_document(file_source: Union[bytes, io.BytesIO, str, Path], filename: str) -> Dict[str, Any]:
    """
    Unified entry point for document parsing.
    Supports .pdf, .docx, .txt.
    Returns metadata, cleaned full text, and per-page chunks.
    """
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        pages = parse_pdf(file_source)
    elif ext in [".docx", ".doc"]:
        pages = parse_docx(file_source)
    elif ext in [".txt", ".md", ".log", ".csv"]:
        pages = parse_txt(file_source)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt")

    if not pages:
        pages = [{
            "page_number": 1,
            "raw_text": "",
            "cleaned_text": "",
            "word_count": 0,
            "char_count": 0
        }]

    full_cleaned_text = "\n\n".join(p["cleaned_text"] for p in pages if p["cleaned_text"])
    full_raw_text = "\n\n".join(p["raw_text"] for p in pages if p["raw_text"])
    
    total_words = sum(p["word_count"] for p in pages)
    total_chars = len(full_cleaned_text)

    return {
        "filename": filename,
        "file_type": ext,
        "page_count": len(pages),
        "total_words": total_words,
        "total_characters": total_chars,
        "cleaned_text": full_cleaned_text,
        "raw_text": full_raw_text,
        "pages": pages
    }
