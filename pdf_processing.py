"""
PDF Processing Module.
Handles text and table extraction from tender PDF documents.
Uses pdfplumber as primary extractor with PyPDF2 as fallback.
"""

import pdfplumber
from PyPDF2 import PdfReader


def extract_text_pdfplumber(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    Better at handling tables and complex layouts common in government tenders.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n--- Page {i + 1} ---\n"
                    full_text += text + "\n"
    except Exception:
        return ""
    return full_text.strip()


def extract_text_pypdf2(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF2.
    Faster but less accurate with tables. Used as fallback.
    """
    full_text = ""
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {i + 1} ---\n"
                full_text += text + "\n"
    except Exception:
        return ""
    return full_text.strip()


def extract_tables(pdf_path: str) -> list:
    """
    Extract tables from PDF using pdfplumber.
    Returns a list of tables, each table being a list of rows.
    """
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
    except Exception:
        pass
    return all_tables


def extract_text(pdf_path: str) -> str:
    """
    Primary extraction function.
    Tries pdfplumber first (better for government tender formats),
    falls back to PyPDF2 if pdfplumber fails.
    """
    text = extract_text_pdfplumber(pdf_path)
    if not text or len(text) < 100:
        text = extract_text_pypdf2(pdf_path)
    return text


def get_page_count(pdf_path: str) -> int:
    """Get the total number of pages in the PDF."""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        return 0
