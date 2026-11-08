"""PDF ingestion: text and table extraction, structure-aware chunking, indexing in ChromaDB."""

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb
import pdfplumber
import pymupdf
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500  # characters, ~400 tokens
MIN_CHUNK_SIZE = 300  # a heading only closes the current chunk past this size
CHUNK_OVERLAP = 150  # ~10 % of CHUNK_SIZE
MAX_HEADING_LENGTH = 80
NO_SECTION = "(aucune)"

DOCS = {
    # key = PDF file name without extension
    "bme280_datasheet": {
        "doc_id": "bme280",
        "heading": r"^[1-9]\d*(\.\d+)*\.?\s+[A-Z][a-z]{2,}(\s|$)",  # "1.3 Pressure sensor specification"
        "skip_pages": {1, 4, 5, 6, 7},  # cover + table of contents
        "dedupe_cells": True
    },
    "samsung_inr18650-35e": {
        "doc_id": "samsung_35e",
        "heading": r"^[1-9]\d*(\.\d+)*\.?\s+[A-Z][a-z]{2,}",  # "7.5 Temperature Dependence..."
        "skip_pages": set(),
        "dedupe_cells": False
    },
    "easa_tcds_falcon7x": {
        "doc_id": "falcon7x_tcds",
        "heading": r"^[IVX]+\.?\s+[A-Z]",  # "II. Certification Basis"
        "skip_pages": {2, 3, 4},  # blank page + table of contents
        "dedupe_cells": False
    },
}

PAGE_NUMBER = re.compile(r"^(Page\s+)?\d+\s*[|/]\s*\d+$")  # "10 | 60", "Page 5/22"
ENDS_WITH_DIGIT = re.compile(r"\d$")  # table-of-contents lines end with a page number
CAPTION_HEIGHT = 40  # points above a table searched for its caption

CHROMA_PATH = "data/chroma"  # git-ignored
COLLECTION_NAME = "doc_chunks"
EMBEDDING_MODEL = "nomic-embed-text"
DOCUMENT_PREFIX = "search_document: "  # nomic-embed-text was trained with these task prefixes
QUERY_PREFIX = "search_query: "

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


@dataclass
class Chunk:
    """A piece of a document, the unit that is indexed, retrieved and cited.

    Attributes:
        doc_id: Short identifier of the source document (e.g. "bme280").
        page: 1-based page number where the chunk is located.
        section: Title of the section the chunk belongs to, or NO_SECTION.
        type: "text" for running text, "table" for a table serialized as markdown.
        id: Stable unique identifier, identical across ingestion runs.
        text: Content of the chunk.
    """

    doc_id: str
    page: int
    section: str
    type: Literal["text", "table"]
    id: str
    text: str


def find_repeated_lines(doc: pymupdf.Document) -> set[str]:
    """Find headers and footers, i.e. lines present on more than half of the pages.

    Args:
        doc: Opened PDF document.

    Returns:
        The stripped repeated lines. Lines of 3 characters or less are excluded so that
        short table values ("0", "25") are never removed.
    """
    counter = Counter()
    for page in doc:
        counter.update({line.strip() for line in page.get_text().splitlines()})
    return {line for line, count in counter.items() if count > len(doc) / 2 and len(line) > 3}


def is_heading(line: str, heading_pattern: re.Pattern) -> bool:
    """Tell whether a line is a section title.

    Args:
        line: Stripped line of text.
        heading_pattern: Document-specific heading regex (see DOCS).

    Returns:
        True if the line matches the pattern, is short, and is not a table-of-contents entry.
    """
    return (
        bool(heading_pattern.match(line))
        and len(line) < MAX_HEADING_LENGTH
        and not ENDS_WITH_DIGIT.search(line)
    )


def make_chunks(text: str, doc_id: str, page: int, section: str, start_index: int) -> list[Chunk]:
    """Split a text into chunks of at most CHUNK_SIZE characters.

    Args:
        text: Text accumulated for one section on one page.
        doc_id: Identifier of the source document.
        page: 1-based page number.
        section: Section title.
        start_index: Number of chunks already created for this document, used to build unique ids.

    Returns:
        The chunks, in reading order.
    """
    return [
        Chunk(
            doc_id=doc_id,
            page=page,
            section=section,
            type="text",
            id=f"{doc_id}_{page}_{start_index + i + 1}",
            text=piece,
        )
        for i, piece in enumerate(text_splitter.split_text(text))
    ]


def extract_text_chunks(pdf_path: str | Path) -> list[Chunk]:
    """Extract the text of a PDF and split it into chunks guided by its section titles.

    A chunk is closed at each new section title (once it reaches MIN_CHUNK_SIZE) and at
    each page end, so that every chunk lies on a single page and can be cited exactly.
    Headers, footers, page numbers and the pages listed in skip_pages are ignored.

    Args:
        pdf_path: Path to a PDF whose file name is a key of DOCS.

    Returns:
        The text chunks of the document, in reading order.

    Raises:
        ValueError: If the PDF has no entry in DOCS.
    """
    stem = Path(pdf_path).stem
    config = DOCS.get(stem)
    if config is None:
        raise ValueError(f"Config not found for {stem}")

    doc_id = config["doc_id"]
    heading_pattern = re.compile(config["heading"])
    chunks: list[Chunk] = []
    buffer: list[str] = []
    section = NO_SECTION
    needs_section = True

    with pymupdf.open(pdf_path) as doc:
        repeated_lines = find_repeated_lines(doc)

        for page_num, page in enumerate(doc, start=1):
            if page_num in config["skip_pages"]:
                continue

            for line in page.get_text().splitlines():
                line = line.strip()
                if not line or PAGE_NUMBER.match(line) or line in repeated_lines:
                    continue

                if is_heading(line, heading_pattern):
                    if sum(len(l) for l in buffer) >= MIN_CHUNK_SIZE:
                        chunks.extend(make_chunks("\n".join(buffer), doc_id, page_num, section, len(chunks)))
                        buffer.clear()
                        needs_section = True
                    if needs_section:
                        section = line
                        needs_section = False

                buffer.append(line)

            if buffer:
                chunks.extend(make_chunks("\n".join(buffer), doc_id, page_num, section, len(chunks)))
                buffer.clear()
                needs_section = True

    return chunks


def clean_row(row: list[str | None], dedupe: bool) -> list[str]:
    """Clean one table row extracted by pdfplumber.

    Args:
        row: Raw cells; None stands for an empty or merged cell.
        dedupe: Drop a cell identical to the previous kept one. Only for documents whose
            merged cells are duplicated by pdfplumber (see DOCS), since elsewhere two equal
            neighbouring cells are genuine values ("97% | 97%").

    Returns:
        The non-empty cells, with line breaks replaced by spaces.
    """
    cleaned: list[str] = []
    for cell in row:
        value = (cell or "").replace("\n", " ").strip()
        if not value:
            continue
        if dedupe and cleaned and value == cleaned[-1]:
            continue
        cleaned.append(value)
    return cleaned


def table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a table to markdown, the first row being the header.

    Args:
        rows: Cleaned rows; they may have different lengths.

    Returns:
        The table as markdown, one line per row.
    """
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    separator = "|" + "---|" * len(rows[0])
    return "\n".join([lines[0], separator, *lines[1:]])


def split_rows(header: list[str], body: list[list[str]], context: str) -> list[list[list[str]]]:
    """Group table rows into pieces that fit in CHUNK_SIZE, repeating the header in each.

    Args:
        header: First row of the table.
        body: Remaining rows.
        context: Text placed before the table in each chunk, counted in the size.

    Returns:
        A list of tables (header + some rows). A single row larger than CHUNK_SIZE is kept whole.
    """
    pieces: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in body:
        candidate = [header, *current, row]
        if current and len(context) + len(table_to_markdown(candidate)) > CHUNK_SIZE:
            pieces.append([header, *current])
            current = []
        current.append(row)
    pieces.append([header, *current])
    return pieces


def find_caption(page: pdfplumber.page.Page, top: float) -> str:
    """Return the last line of text just above a table, usually its caption.

    Args:
        page: pdfplumber page containing the table.
        top: Vertical position of the top of the table.

    Returns:
        The caption ("Table 3: Pressure parameter specification"), or "" if nothing is above.
    """
    if top < 1:
        return ""
    above = page.crop((0, max(0, top - CAPTION_HEIGHT), page.width, top)).extract_text() or ""
    lines = above.strip().splitlines()
    return lines[-1].strip() if lines else ""


def extract_table_chunks(pdf_path: str | Path, page_sections: dict[int, str]) -> list[Chunk]:
    """Extract the tables of a PDF as markdown chunks.

    Each chunk starts with the table caption and section, to give context to the embedding,
    followed by the table. Tables with fewer than 2 rows or 2 columns are ignored.

    Args:
        pdf_path: Path to a PDF whose file name is a key of DOCS.
        page_sections: Section title per page, taken from the text chunks.

    Returns:
        The table chunks of the document, in reading order.

    Raises:
        ValueError: If the PDF has no entry in DOCS.
    """
    stem = Path(pdf_path).stem
    config = DOCS.get(stem)
    if config is None:
        raise ValueError(f"Config not found for {stem}")

    doc_id = config["doc_id"]
    chunks: list[Chunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num in config["skip_pages"]:
                continue

            for table in page.find_tables():
                rows = [clean_row(row, config["dedupe_cells"]) for row in table.extract()]
                rows = [row for row in rows if row]
                if len(rows) < 2 or max(len(row) for row in rows) < 2:
                    continue

                section = page_sections.get(page_num, NO_SECTION)
                caption = find_caption(page, table.bbox[1])
                context = f"{caption}\n(section: {section})\n\n" if caption else f"(section: {section})\n\n"

                for piece in split_rows(rows[0], rows[1:], context):
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        page=page_num,
                        section=section,
                        type="table",
                        id=f"{doc_id}_{page_num}_t{len(chunks) + 1}",
                        text=context + table_to_markdown(piece),
                    ))

    return chunks

def extract_chunks(pdf_path: str | Path) -> list[Chunk]:
    """Extract all chunks of a PDF: text first, then tables.

    Args:
        pdf_path: Path to a PDF whose file name is a key of DOCS.

    Returns:
        The text and table chunks of the document.
    """
    text_chunks = extract_text_chunks(pdf_path)
    page_sections = {c.page: c.section for c in text_chunks}
    return text_chunks + extract_table_chunks(pdf_path, page_sections)


def get_collection() -> chromadb.Collection:
    """Open the persisted collection of indexed chunks.

    Returns:
        The ChromaDB collection.

    Raises:
        chromadb.errors.NotFoundError: If nothing has been ingested yet.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed the chunks and store them in a freshly rebuilt ChromaDB collection.

    The collection is deleted and recreated on every call: an upsert would leave behind
    chunks that no longer exist when the chunking settings change.

    Args:
        chunks: All chunks of the corpus.

    Returns:
        The number of chunks stored in the collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})

    vectors = embeddings.embed_documents([DOCUMENT_PREFIX + c.text for c in chunks])
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"doc_id": c.doc_id, "page": c.page, "section": c.section, "type": c.type} for c in chunks],
    )
    return collection.count()


def ingest_all(pdf_dir: str | Path = "data/pdfs") -> list[Chunk]:
    """Extract the chunks of every PDF of a folder and index them.

    Args:
        pdf_dir: Folder containing the corpus PDFs.

    Returns:
        All indexed chunks.

    Raises:
        FileNotFoundError: If the folder contains no PDF.
    """
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {pdf_dir}")
    chunks = [chunk for pdf in pdfs for chunk in extract_chunks(pdf)]
    index_chunks(chunks)
    return chunks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
        chunks = extract_chunks(pdf)
        tables = [c for c in chunks if c.type == "table"]
        print(f"{pdf.name}: {len(chunks) - len(tables)} text chunks, {len(tables)} tables")
        for t in tables[:2]:
            print(t.id, "|", t.section)
            print(t.text)
            print()
