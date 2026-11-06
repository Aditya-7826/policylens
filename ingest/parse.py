"""Turn PDF / HTML / TXT files into plain text pages, keeping page numbers.

Output is a `Document` with a list of `Page`s. HTML and TXT have no real pages,
so they come back as a single page with number None.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pypdf import PdfReader

SUPPORTED = {".pdf", ".html", ".htm", ".txt"}

# Friendly titles for the known corpus; anything else gets a title from its filename.
KNOWN_TITLES = {
    "dpdp_act_2023": "DPDP Act 2023",
    "gdpr": "GDPR",
    "nist_csf_2.0": "NIST CSF 2.0",
    "nist_sp_800-53r5": "NIST SP 800-53r5",
}


@dataclass
class Page:
    number: int | None
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    path: Path
    pages: list[Page] = field(default_factory=list)
    kind: str = "txt"  # pdf | gazette | html | txt - picks the chunking rules


def doc_id_for(path: Path) -> str:
    stem = path.stem.lower()
    return re.sub(r"[^a-z0-9._-]+", "_", stem).strip("_")


def clean_text(text: str) -> str:
    """Normalise unicode, fix hyphenation at line ends, collapse spaces (keep newlines)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split across lines
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> tuple[list[Page], str]:
    """Returns (pages, kind). Gazette of India PDFs get a layout-aware parser."""
    reader = PdfReader(str(path))
    head = " ".join((reader.pages[i].extract_text() or "") for i in range(min(3, len(reader.pages))))
    if "GAZETTE OF INDIA" in head.upper():
        return parse_gazette_pdf(path), "gazette"
    pages = [Page(i + 1, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return strip_running_headers(pages), "pdf"


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line)).strip().lower()


def strip_running_headers(pages: list[Page], zone: int = 4, min_share: float = 0.3) -> list[Page]:
    """Drop running headers/footers: lines near the top or bottom of a page that repeat
    (digits ignored, so "PAGE 34" == "PAGE 35") on at least `min_share` of pages."""
    from collections import Counter

    if len(pages) < 5:
        return pages
    counts: Counter[str] = Counter()
    for p in pages:
        lines = [ln for ln in p.text.split("\n") if ln.strip()]
        counts.update({_norm_line(ln) for ln in lines[:zone] + lines[-zone:]})
    threshold = max(3, int(len(pages) * min_share))
    repeated = {k for k, c in counts.items() if c >= threshold}
    out = []
    for p in pages:
        lines = p.text.split("\n")
        keep = []
        for i, ln in enumerate(lines):
            near_edge = i < zone + 2 or i >= len(lines) - zone - 2
            norm = _norm_line(ln)
            if near_edge and (norm in repeated or re.fullmatch(r"[#\s]*", norm) or set(norm) <= {"_", " "}):
                continue
            keep.append(ln)
        out.append(Page(p.number, "\n".join(keep).strip()))
    return out


# ---------------------------------------------------------------- Gazette of India layout
# Indian Acts are printed with section titles as *margin notes* (left margin on even
# pages, right margin on odd pages) and a running header at the top. Plain text
# extraction mixes all of that into the body, so we split each page into columns:
#   body   = words with 100 <= x0 <= 478
#   margin = everything else -> section titles ("Short title and commencement.")
# and inject a marker line "Section 8: General obligations of Data Fiduciary." before
# each section's first line. Margin notes like "24 of 1997." are Act citations, not titles.
GAZETTE_TOP, GAZETTE_BOTTOM = 87, 787  # points from the top: skip header and footer
MARGIN_LEFT, MARGIN_RIGHT = 100, 478
_ACT_CITATION = re.compile(r"^\d+ of \d{4}\.?$")
_SECTION_START = re.compile(r"^(\d{1,3})\.\s")


def _group_margin_notes(words: list[dict]) -> list[tuple[float, str]]:
    """Group margin words into lines, then lines into notes (a note ends with a full stop)."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) < 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    notes: list[tuple[float, str]] = []
    open_note = False
    for line in lines:
        text = " ".join(w["text"] for w in line)
        if _ACT_CITATION.match(text):
            open_note = False
            continue
        if open_note and line[0]["top"] - notes[-1][2] < 14:
            notes[-1] = (notes[-1][0], notes[-1][1] + " " + text, line[0]["top"])
        else:
            notes.append((line[0]["top"], text, line[0]["top"]))
        open_note = not text.endswith(".")
    return [(top, text) for top, text, _ in notes]


def _schedule_rows(band, words: list[dict]) -> list[str]:
    """The penalty Schedule is a 2-column table; reading it line by line interleaves the columns
    ("May extend to two take reasonable ... hundred and fifty"). Rebuild one line per row:
    rows start at a serial number ("1.") on the left; the penalty column starts at the x of the
    "Penalty" header."""
    header = next((w for w in words if w["text"] == "Penalty"), None)
    serials = [w for w in words if re.fullmatch(r"\d{1,2}\.", w["text"]) and w["x0"] < 160]
    if header is None or not serials:
        return ["Schedule: The Schedule", band.extract_text() or ""]
    # Work with whole words (cropping by characters splits words that straddle the column edge).
    # pdfplumber keeps absolute page coordinates on cropped pages.
    x0, top, x1, _ = band.bbox
    intro = band.crop((x0, top, x1, serials[0]["top"] - 1)).extract_text() or ""
    lines = ["Schedule: The Schedule", intro.replace("THE SCHEDULE", "").strip()]
    end = next((w["top"] for w in words if w["text"].startswith("——")), float("inf"))
    tops = [w["top"] - 1 for w in serials] + [end]
    for i, serial in enumerate(serials):
        row = sorted((w for w in words if tops[i] <= w["top"] < tops[i + 1]), key=lambda w: (round(w["top"]), w["x0"]))
        first = [w for w in row if abs(w["top"] - serial["top"]) < 3 and w is not serial]
        # column boundary = middle of the widest gap between words on the row's first line
        gaps = [(b["x0"] - a["x1"], (a["x1"] + b["x0"]) / 2) for a, b in zip(first, first[1:])]
        split_x = max(gaps)[1] if gaps else header["x0"]
        desc = " ".join(w["text"] for w in row if w["x0"] < split_x and w is not serial)
        pen = " ".join(w["text"] for w in row if w["x0"] >= split_x)
        lines.append(f"{serial['text']} {desc} Penalty: {pen}")
    return lines


def parse_gazette_pdf(path: Path) -> list[Page]:
    import pdfplumber

    pages: list[Page] = []
    in_schedule = False  # numbered rows in "THE SCHEDULE" are table rows, not sections
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            top = GAZETTE_TOP
            if p.page_number == 1:  # skip the masthead: start at the Act's title line
                hits = p.search(r"ACT, \d{4}")
                if hits:
                    top = hits[0]["top"] - 2
            band = p.crop((0, top, p.width, GAZETTE_BOTTOM))
            words = band.extract_words()
            if in_schedule or any(w["text"] == "SCHEDULE" for w in words):
                in_schedule = True
                pages.append(Page(p.page_number, clean_text("\n".join(_schedule_rows(band, words)))))
                continue
            margin = [w for w in words if w["x0"] < MARGIN_LEFT or w["x0"] > MARGIN_RIGHT]
            boxes = [(w["x0"] - 0.5, w["x1"] + 0.5, w["top"] - 1, w["bottom"] + 1) for w in margin]

            def in_body(obj, boxes=boxes):
                if obj.get("object_type") != "char":
                    return True
                return not any(x0 <= obj["x0"] and obj["x1"] <= x1 and t <= obj["top"] <= b
                               for x0, x1, t, b in boxes)

            body = band.filter(in_body)
            notes = _group_margin_notes(margin)
            out: list[str] = []
            for line in body.extract_text_lines():
                if line["text"].strip().upper() in ("THE SCHEDULE", "SCHEDULE"):
                    in_schedule = True
                    out.append("Schedule: The Schedule")
                m = None if in_schedule else _SECTION_START.match(line["text"])
                if m:
                    near = [n for n in notes if abs(n[0] - line["top"]) < 25]
                    title = min(near, key=lambda n: abs(n[0] - line["top"]))[1] if near else ""
                    out.append(f"Section {m.group(1)}: {title}".rstrip(": "))
                out.append(line["text"])
            pages.append(Page(p.page_number, clean_text("\n".join(out))))
    return pages


BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "div", "br", "tr"]


def html_to_text(html: str | bytes) -> str:
    with warnings.catch_warnings():  # EUR-Lex serves XHTML; the HTML parser handles it fine
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    # Put a newline after each block-level element so headings stay on their own line.
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    return clean_text(soup.get_text())


def parse_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    kind = suffix.lstrip(".")
    if suffix == ".pdf":
        pages, kind = parse_pdf(path)
    elif suffix in (".html", ".htm"):
        pages, kind = [Page(None, html_to_text(path.read_bytes()))], "html"
    elif suffix == ".txt":
        pages = [Page(None, clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    else:
        raise ValueError(f"unsupported file type: {path.name}")
    doc_id = doc_id_for(path)
    title = KNOWN_TITLES.get(doc_id, path.stem.replace("_", " ").replace("-", " ").title())
    return Document(doc_id, title, path, [p for p in pages if p.text], kind)


def discover(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in SUPPORTED and p.is_file())
