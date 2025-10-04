from pathlib import Path
import fitz
import os


def extract_text_from_file(file_path: Path):
    """
    Extract text from a file. Currently supports .txt and .pdf
    """
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif file_path.suffix.lower() == ".pdf":
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    else:
        print(f"Unsupported file type: {file_path.suffix}")
        return ""


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


def digest_directory(directory: Path):
    """
    Recursively extract text from all files in a directory and split into chunks.
    """
    all_chunks = []
    for file_path in directory.rglob("*"):
        print(f"Processing file: {file_path}")
        if file_path.is_file():
            text = extract_text_from_file(file_path)
            if text:
                chunks = split_text_into_chunks(text)
                all_chunks.extend(chunks)

        print(f"Processing file done: {file_path}")
    return all_chunks
