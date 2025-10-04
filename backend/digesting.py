from pathlib import Path
import fitz
import os


def extract_text_from_file(file_path: Path, chunk_by_page: bool = False):
    """
    Extract text from a file. Supports .txt and .pdf.
    If chunk_by_page is True for PDFs, each page is returned as a separate chunk.
    """
    if file_path.suffix.lower() == ".txt":
        return [file_path.read_text(encoding="utf-8")]

    elif file_path.suffix.lower() == ".pdf":
        chunks = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text()
                if chunk_by_page:
                    chunks.append(page_text)
                else:
                    chunks.append(page_text)  # Will be concatenated later if needed
        if not chunk_by_page:
            return ["".join(chunks)]
        return chunks

    else:
        print(f"Unsupported file type: {file_path.suffix}")
        return []


def split_text_into_chunks(text: str, chunk_size: int = 100, overlap: int = 10):
    """
    Split text into smaller chunks for processing.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def digest_directory(directory: Path, chunk_pdf_by_page: bool = True, text_chunk_size: int = 500, text_overlap: int = 25):
    """
    Recursively extract text from all files in a directory and split into chunks.
    For PDFs, can treat each page as a separate chunk.
    """
    all_chunks = []
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            print(f"Processing file: {file_path}")
            chunks = extract_text_from_file(file_path, chunk_by_page=chunk_pdf_by_page)
            if chunks:
                for chunk in chunks:
                    # Only split further if it's from a txt file or if not chunking PDF by page
                    if file_path.suffix.lower() == ".txt" or not chunk_pdf_by_page:
                        all_chunks.extend(split_text_into_chunks(chunk, chunk_size=text_chunk_size, overlap=text_overlap))
                    else:
                        all_chunks.append(chunk)
            print(f"Processing file done: {file_path}")
    return all_chunks
