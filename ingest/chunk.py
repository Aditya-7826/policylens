"""Structure-aware chunking.

Step 1: split each document into its legal/standard units (Article, Section, Control...)
        using per-document heading rules (`PROFILES`), with a generic fallback.
Step 2: split units that are too long into ~max_tokens windows on paragraph
        boundaries, with a small token overlap.

Chunk IDs are readable and stable, e.g. `gdpr::article-33::0`, so a citation like
[GDPR, Article 33, gdpr::article-33::0] can be checked by a human.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken
from pydantic import BaseModel

from ingest.parse import Document

_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str  # "Article 33", "Section 8", "AC-2"
    section_title: str  # "Notification of a personal data breach..."
    page: int | None
    text: str
    n_tokens: int
    content_hash: str

    @property
    def header(self) -> str:
        title = f": {self.section_title}" if self.section_title else ""
        return f"{self.doc_title} - {self.section}{title}"

    @property
    def context_text(self) -> str:
        """Text used for embeddings, BM25 and LLM prompts: header gives the chunk its context."""
        return f"{self.header}\n{self.text}"

    def citation(self) -> str:
        return f"[{self.doc_title}, {self.section}, {self.chunk_id}]"


@dataclass
class HeadingRule:
    """A line matching `pattern` starts a new unit. Group 1 = unit label, group 2 (optional) = title.

    If `title_next_line` is set and group 2 is empty, the next non-empty line is used as the title.
    """

    pattern: re.Pattern
    label: str = "{0}"  # format string applied to group 1
    title_next_line: bool = False
    until: str | None = None  # rule switches off once a unit whose label starts with this exists


def _r(p: str) -> re.Pattern:
    return re.compile(p)


# Marker lines injected by the Gazette parser: "Section 8: General obligations of Data Fiduciary."
GAZETTE_RULE = HeadingRule(_r(r"^(Section \d{1,3})(?::\s*(.*))?$"))

GENERIC_RULES = [
    HeadingRule(_r(r"^(Article\s+\d+[a-z]?)\s*$"), title_next_line=True),
    HeadingRule(_r(r"^(Section\s+\d+[A-Z]?)[.:\s-]+(.*)$")),
    HeadingRule(_r(r"^(\d{1,3}(?:\.\d{1,3}){0,2})\s+([A-Z][A-Za-z ,'&/-]{3,80})$"), label="Section {0}"),
]

# Rules by doc_id first, then by document kind (see ingest.parse.Document.kind).
PROFILES: dict[str, list[HeadingRule]] = {
    "gazette": [GAZETTE_RULE, HeadingRule(_r(r"^(Schedule): (.*)$"))],
    "gdpr": [
        # Recitals "(85)" alone on a line, only in the preamble (before Article 1).
        HeadingRule(_r(r"^\((\d{1,3})\)$"), label="Recital {0}", until="Article"),
        HeadingRule(_r(r"^(Article\s+\d+)\s*$"), title_next_line=True),
    ],
    "nist_sp_800-53r5": [
        # Controls: "AC-2 ACCOUNT MANAGEMENT" (all caps; appendix tables use mixed case and are skipped)
        HeadingRule(_r(r"^([A-Z]{2}-\d{1,2})\s+([A-Z][A-Z0-9 ,/&()—\-]{3,90})$")),
        # Narrative chapters: "2.1 REQUIREMENTS AND CONTROLS"
        HeadingRule(_r(r"^(\d\.\d{1,2})\s+([A-Z][A-Z ,/&\-]{3,80})$"), label="Section {0}"),
    ],
    "nist_csf_2.0": [
        # Functions: "GOVERN (GV): The organization's ..."
        HeadingRule(_r(r"^(?P<title>GOVERN|IDENTIFY|PROTECT|DETECT|RESPOND|RECOVER) \((?P<label>[A-Z]{2})\):")),
        # Categories: "• Organizational Context (GV.OC): The circumstances ..."
        HeadingRule(_r(r"^•\s*(?P<title>[A-Z][A-Za-z ,&\-]{3,60}) \((?P<label>[A-Z]{2}\.[A-Z]{2})\):")),
        # Narrative sections: "2. Introduction to the CSF Core" (2+ real words, so figure labels don't match)
        HeadingRule(_r(r"^(\d(?:\.\d)?)\.?\s+([A-Z][a-z]+(?: [A-Za-z(),'&/\-]+){1,12})$"), label="Section {0}"),
    ],
}


@dataclass
class Unit:
    section: str
    title: str
    page: int | None
    lines: list[str]


def split_units(doc: Document, rules: list[HeadingRule] | None = None) -> list[Unit]:
    if rules is None:
        rules = PROFILES.get(doc.doc_id) or PROFILES.get(doc.kind) or GENERIC_RULES
    first_page = doc.pages[0].number if doc.pages else None
    units: list[Unit] = [Unit("Preamble", "", first_page, [])]
    want_title = False
    disabled: set[int] = set()
    for page in doc.pages:
        for raw in page.text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if want_title:
                units[-1].title = line[:160]
                want_title = False
                continue
            match_rule = None
            for idx, rule in enumerate(rules):
                if idx in disabled:
                    continue
                m = rule.pattern.match(line)
                if m:
                    match_rule = (rule, m)
                    break
            if match_rule:
                rule, m = match_rule
                groups = m.groupdict()
                raw_label = groups.get("label") or m.group(1)
                if "title" in groups:
                    title = groups["title"] or ""
                else:
                    title = (m.group(2) or "") if m.lastindex and m.lastindex >= 2 else ""
                label = rule.label.format(raw_label.strip())
                for idx, other in enumerate(rules):
                    if other.until and label.startswith(other.until):
                        disabled.add(idx)
                title = title.strip()
                units.append(Unit(label, title.title() if title.isupper() else title, page.number, []))
                # Keep the heading line when it carries body text (e.g. a CSF category description).
                if line[m.end():].strip():
                    units[-1].lines.append(line)
                want_title = rule.title_next_line and not title
                continue
            units[-1].lines.append(line)
    return [u for u in units if u.lines]


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split a paragraph longer than max_tokens by sentences, then hard-split by tokens."""
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.;:])\s+", text):
        candidate = f"{cur} {sent}".strip()
        if n_tokens(candidate) <= max_tokens:
            cur = candidate
            continue
        if cur:
            pieces.append(cur)
        if n_tokens(sent) <= max_tokens:
            cur = sent
        else:
            ids = _enc.encode(sent, disallowed_special=())
            pieces.extend(_enc.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens))
            cur = ""
    if cur:
        pieces.append(cur)
    return pieces


def window(paragraphs: list[str], max_tokens: int, overlap: int) -> list[str]:
    """Greedy pack paragraphs into windows of <= max_tokens; carry ~overlap tokens forward."""
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, max_tokens) if n_tokens(p) > max_tokens else [p])
    windows, cur, cur_tok = [], [], 0
    for u in units:
        t = n_tokens(u)
        if cur and cur_tok + t > max_tokens:
            windows.append("\n".join(cur))
            # overlap: keep trailing paragraphs that fit in `overlap` tokens
            tail, tail_tok = [], 0
            for prev in reversed(cur):
                pt = n_tokens(prev)
                if tail_tok + pt > overlap:
                    break
                tail.insert(0, prev)
                tail_tok += pt
            if tail_tok + t > max_tokens:  # overlap must never push a window over the cap
                tail, tail_tok = [], 0
            cur, cur_tok = tail, tail_tok
        cur.append(u)
        cur_tok += t
    if cur:
        windows.append("\n".join(cur))
    return windows


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", label.lower()).strip("-")


def _paragraphs(lines: list[str]) -> list[str]:
    """PDF text breaks lines mid-sentence; re-join lines into paragraphs at clause starts."""
    paras: list[str] = []
    for line in lines:
        starts_new = (not paras or re.match(r"^(\(\w{1,4}\)|\d+\.|[a-z]\.|•|-|Article|Section)\s", line)
                      or paras[-1].endswith((".", ":", ";")))
        if starts_new:
            paras.append(line)
        else:
            paras[-1] += " " + line
    return paras


def chunk_document(doc: Document, max_tokens: int = 700, overlap: int = 80,
                   rules: list[HeadingRule] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for unit in split_units(doc, rules):
        slug = _slug(unit.section)
        # The same label can appear twice (e.g. a table of contents); keep IDs unique and stable.
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        for i, text in enumerate(window(_paragraphs(unit.lines), max_tokens, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{slug}::{i}",
                doc_id=doc.doc_id, doc_title=doc.title,
                section=unit.section, section_title=unit.title, page=unit.page,
                text=text, n_tokens=n_tokens(text),
                content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
            ))
    return chunks
