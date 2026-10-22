from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import fitz
import pytesseract
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image


AUTHORITY_TYPES = {"policy", "addendum", "memo", "handbook", "calendar"}
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"),
    re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b", re.I),
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\.?(?:,?\s+(20\d{{2}}))?\b", re.I),
)
HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+.{2,100}|[A-Z][A-Z \-]{4,70})$")


@dataclass(frozen=True)
class TextBlock:
    text: str
    locator: str
    heading: bool = False


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    locator: str
    text: str


def _pdf_blocks(path: Path) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 40:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                text = pytesseract.image_to_string(image).strip()
            if not text:
                raise RuntimeError(f"OCR produced no text for scanned page {page_number}")
            locator = f"p. {page_number}"
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                is_heading = bool(HEADING_RE.match(stripped))
                blocks.append(TextBlock(stripped, locator, is_heading))
    return blocks


def _docx_blocks(path: Path) -> list[TextBlock]:
    document = Document(path)
    blocks: list[TextBlock] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            style = paragraph.style.name.lower() if paragraph.style else ""
            blocks.append(TextBlock(text, "", style.startswith("heading")))
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " / ") for cell in row.cells]
            if any(cells):
                blocks.append(TextBlock(" | ".join(cells), "Table"))
    return blocks


def _html_blocks(path: Path) -> list[TextBlock]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    blocks: list[TextBlock] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if element.name == "tr":
            text = " | ".join(
                cell.get_text(" ", strip=True) for cell in element.find_all(["th", "td"])
            )
            if text:
                blocks.append(TextBlock(text, "Table"))
        else:
            text = element.get_text(" ", strip=True)
            if text:
                blocks.append(TextBlock(text, "", element.name.startswith("h")))
    return blocks


def extract_blocks(path: Path) -> list[TextBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_blocks(path)
    if suffix == ".docx":
        return _docx_blocks(path)
    if suffix == ".html":
        return _html_blocks(path)
    if suffix in {".md", ".markdown", ".txt"}:
        blocks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                blocks.append(TextBlock(line.lstrip("# ").strip(), "", line.startswith("#")))
        return blocks
    raise ValueError(f"Unsupported corpus format: {suffix}")


def _split_text(text: str, limit: int = 1800) -> Iterable[str]:
    lines = text.splitlines()
    current: list[str] = []
    size = 0
    for line in lines:
        pieces = [line]
        while any(len(part) > limit for part in pieces):
            long_part = next(part for part in pieces if len(part) > limit)
            index = pieces.index(long_part)
            pieces[index:index + 1] = [
                long_part[i:i + limit] for i in range(0, len(long_part), limit)
            ]
        for piece in pieces:
            if current and size + len(piece) + 1 > limit:
                yield "\n".join(current)
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        yield "\n".join(current)


def chunk_document(doc_id: str, blocks: list[TextBlock]) -> list[Chunk]:
    has_headings = any(block.heading for block in blocks)
    if not has_headings:
        by_page: dict[str, list[str]] = {}
        order: list[str] = []
        for block in blocks:
            locator = block.locator or "Document"
            if locator not in by_page:
                by_page[locator] = []
                order.append(locator)
            by_page[locator].append(block.text)
        return [
            Chunk(doc_id, locator, part)
            for locator in order
            for part in _split_text("\n".join(by_page[locator]))
        ]
    sections: list[tuple[str, str]] = []
    heading = ""
    page = ""
    current: list[str] = []
    for block in blocks:
        if block.heading:
            if current:
                sections.append((heading or page or "Document", "\n".join(current)))
                current = []
            heading = block.text
        if block.locator and block.locator.startswith("p. "):
            page = block.locator
        if not block.heading:
            current.append(block.text)
    if current:
        sections.append((heading or page or "Document", "\n".join(current)))
    chunks: list[Chunk] = []
    for locator, text in sections:
        chunks.extend(Chunk(doc_id, locator, part) for part in _split_text(text))
    return chunks


def _date_candidates(text: str, default_year: int) -> set[str]:
    holidays: set[str] = set()
    for line in text.splitlines():
        if re.search(r"effective date|version|document id", line, re.I):
            continue
        # A date is a holiday only when its row/line also names an event.
        if not re.search(r"[A-Za-z]{3,}", line):
            continue
        for index, pattern in enumerate(DATE_PATTERNS):
            for match in pattern.finditer(line):
                try:
                    if index == 0:
                        year, month, day = map(int, match.groups())
                    elif index == 1:
                        month_name, day_text, year_text = match.groups()
                        month = _month_number(month_name)
                        day = int(day_text)
                        year = int(year_text or default_year)
                    else:
                        day_text, month_name, year_text = match.groups()
                        month = _month_number(month_name)
                        day = int(day_text)
                        year = int(year_text or default_year)
                    holidays.add(date(year, month, day).isoformat())
                except (ValueError, KeyError):
                    continue
    return holidays


def _month_number(name: str) -> int:
    from datetime import datetime

    return datetime.strptime(name[:3].title(), "%b").month


def fingerprint(manifest_path: Path, corpus_dir: Path) -> str:
    digest = hashlib.sha256(manifest_path.read_bytes())
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _init_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, doc_type TEXT NOT NULL,
            regions TEXT NOT NULL, effective_date TEXT NOT NULL, access_level TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            doc_id UNINDEXED, locator UNINDEXED, content,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE IF NOT EXISTS holidays(
            doc_id TEXT NOT NULL, region TEXT NOT NULL, holiday_date TEXT NOT NULL,
            PRIMARY KEY(doc_id, region, holiday_date)
        );
        """
    )


def build_index(manifest_path: Path, corpus_dir: Path, index_path: Path) -> str:
    if not manifest_path.is_file() or not corpus_dir.is_dir():
        raise FileNotFoundError("Corpus directory or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest.json must contain a non-empty document array")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    source_fingerprint = fingerprint(manifest_path, corpus_dir)
    connection = sqlite3.connect(index_path)
    try:
        connection.row_factory = sqlite3.Row
        _init_index(connection)
        previous = connection.execute(
            "SELECT value FROM index_meta WHERE key = 'fingerprint'"
        ).fetchone()
        if previous and previous["value"] == source_fingerprint:
            count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            if count == len(manifest):
                return source_fingerprint
        prepared: list[tuple[dict[str, object], list[Chunk], set[str]]] = []
        for item in manifest:
            required = {"doc_id", "file", "title", "doc_type", "regions", "effective_date", "access_level"}
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError("Manifest entry is missing required metadata")
            file_name = Path(str(item["file"])).name
            if file_name != item["file"]:
                raise ValueError("Manifest file entries must be corpus-local names")
            blocks = extract_blocks(corpus_dir / file_name)
            chunks = chunk_document(str(item["doc_id"]), blocks)
            if not chunks:
                raise ValueError(f"No text extracted from {file_name}")
            full_text = "\n".join(chunk.text for chunk in chunks)
            holidays = (
                _date_candidates(full_text, int(str(item["effective_date"])[:4]))
                if item["doc_type"] == "calendar"
                else set()
            )
            prepared.append((item, chunks, holidays))
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
        connection.execute("DELETE FROM holidays")
        for item, chunks, holidays in prepared:
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item["doc_id"],
                    item["title"],
                    item["doc_type"],
                    json.dumps(item["regions"]),
                    item["effective_date"],
                    item["access_level"],
                ),
            )
            connection.executemany(
                "INSERT INTO chunks(doc_id, locator, content) VALUES (?, ?, ?)",
                [(chunk.doc_id, chunk.locator, chunk.text) for chunk in chunks],
            )
            for region in item["regions"]:
                if region != "GLOBAL":
                    connection.executemany(
                        "INSERT OR IGNORE INTO holidays VALUES (?, ?, ?)",
                        [(item["doc_id"], region, day) for day in holidays],
                    )
        connection.execute(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES ('fingerprint', ?)",
            (source_fingerprint,),
        )
        connection.commit()
        return source_fingerprint
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
