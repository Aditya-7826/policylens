"""Knowledge-graph extraction: schema, spaCy pre-filter, LLM extraction, entity resolution.

Everything here is pure Python (no database). It turns one chunk into a `ChunkGraph`
(nodes + edges). `ingest.graph` writes ChunkGraphs into Neo4j.

Graph model
-----------
  (:Regulation)-[:CONTAINS]->(:Section)                      deterministic
  (:Section)-[:REQUIRES]->(:Obligation)                      deterministic (from the chunk's section)
  (:Section)-[:DEFINES]->(:Definition)                       deterministic
  (:Section)-[:CONTAINS]->(:Right|:Penalty)                  deterministic
  (:Section)-[:REFERENCES]->(:Section)                       deterministic (regex on "Article 55", "AC-2", ...)
  (:Obligation|:Right)-[:APPLIES_TO]->(:Entity)              LLM
  (:Penalty)-[:PENALIZES]->(:Obligation|:Entity)             LLM
  (:Definition)-[:DEFINES]->(:Entity)                        LLM
  (:Obligation)-[:REQUIRES]->(:Obligation)                   LLM (one duty triggers another)
  (x)-[:SIMILAR_TO]-(y)  across regulations                  local embeddings (ingest.graph)
  every node -[:MENTIONED_IN]->(:Chunk)                      provenance for citations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ingest.chunk import Chunk

NodeType = Literal["Obligation", "Entity", "Right", "Penalty", "Definition"]
RelType = Literal["APPLIES_TO", "PENALIZES", "DEFINES", "REQUIRES"]

# (source type, relation, target type) combinations the LLM may produce. Anything else is dropped.
ALLOWED_TRIPLES = {
    ("Obligation", "APPLIES_TO", "Entity"),
    ("Right", "APPLIES_TO", "Entity"),
    ("Penalty", "PENALIZES", "Obligation"),
    ("Penalty", "PENALIZES", "Entity"),
    ("Definition", "DEFINES", "Entity"),
    ("Obligation", "REQUIRES", "Obligation"),
}
SECTION_REL_FOR = {"Obligation": "REQUIRES", "Definition": "DEFINES", "Right": "CONTAINS", "Penalty": "CONTAINS"}


# ----------------------------------------------------------------------------- LLM schema
class ExtractedNode(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    type: NodeType
    description: str = Field(default="", max_length=400)

    @field_validator("type", mode="before")
    @classmethod
    def _cap(cls, v):
        return v.strip().title() if isinstance(v, str) else v


class ExtractedRelation(BaseModel):
    source: str
    relation: RelType
    target: str

    @field_validator("relation", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper().replace(" ", "_") if isinstance(v, str) else v


class Extraction(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("nodes", "relations", mode="before")
    @classmethod
    def _drop_invalid(cls, items, info):
        """Keep the valid items instead of rejecting the whole extraction for one bad item."""
        if not isinstance(items, list):
            return []
        model = ExtractedNode if info.field_name == "nodes" else ExtractedRelation
        good = []
        for it in items:
            try:
                good.append(model.model_validate(it))
            except Exception:
                continue
        return good


EXTRACTION_SYSTEM = """You extract a compliance knowledge graph from one passage of a regulation or standard.
Return ONLY a JSON object: {"nodes": [...], "relations": [...]}.

Node types (use exactly these):
- Obligation: a duty someone must perform (e.g. "notify personal data breach to the Board").
- Entity: a role or actor (e.g. "Data Fiduciary", "Controller", "Data Protection Board").
- Right: a right held by someone (e.g. "right to erasure").
- Penalty: a fine or sanction (e.g. "penalty up to 250 crore rupees").
- Definition: a defined term (e.g. "personal data breach").
Each node: {"name": short canonical name, "type": ..., "description": one sentence from the passage}.

Relations (source and target are node names from your list):
- Obligation APPLIES_TO Entity (who must do it); Right APPLIES_TO Entity (who holds it)
- Penalty PENALIZES Obligation or Entity
- Definition DEFINES Entity
- Obligation REQUIRES Obligation (one duty triggers another)
Each relation: {"source": name, "relation": ..., "target": name}.

Rules: use only facts stated in the passage; at most 8 nodes; names in the passage's own terms;
if the passage has nothing relevant return {"nodes": [], "relations": []}."""


def extraction_messages(chunk: Chunk) -> list[dict]:
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Source: {chunk.header}\n\nPassage:\n{chunk.text}"},
    ]


# ------------------------------------------------------------------------ spaCy pre-filter
SIGNAL_LEMMAS = {
    "shall", "must", "may", "should", "mean", "require", "ensure", "right", "penalty", "fine",
    "obligation", "notify", "implement", "define", "establish", "prohibit", "liable", "comply",
    "document", "protect", "monitor", "review", "erase", "consent",
}


@lru_cache
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


def worth_extracting(chunk: Chunk, min_words: int = 25) -> tuple[bool, str]:
    """Cheap local check so obviously empty chunks never reach the LLM."""
    text = chunk.text
    if re.search(r"\[Withdrawn:", text) and len(text) < 400:
        return False, "withdrawn control"
    doc = _nlp()(text[:4000])
    words = [t for t in doc if t.is_alpha and not t.is_stop]
    if len(words) < min_words:
        return False, f"only {len(words)} content words"
    if not any(t.tag_ == "MD" or t.lemma_.lower() in SIGNAL_LEMMAS for t in doc):
        return False, "no normative language"
    return True, "ok"


# ------------------------------------------------------------------- entity resolution
_ARTICLES = re.compile(r"^(the|a|an|any|such|every|each)\s+")


def normalize_name(name: str) -> str:
    """'The Data Fiduciaries' -> 'data fiduciary'. Used to merge duplicate entities."""
    n = name.lower().strip()
    n = re.sub(r"[\"'“”‘’`]", "", n)
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = _ARTICLES.sub("", n)
    words = n.split()
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 4:
            words[-1] = last[:-3] + "y"
        elif last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 3:
            words[-1] = last[:-1]
    return " ".join(words)


def node_key(node_type: str, name: str, doc_id: str) -> str:
    """Entities are shared across regulations ("processor"); everything else is scoped to its regulation."""
    norm = normalize_name(name)
    return f"Entity:{norm}" if node_type == "Entity" else f"{node_type}:{doc_id}:{norm}"


def section_key(doc_id: str, section: str) -> str:
    return f"{doc_id}::{section}"


# --------------------------------------------------------------- deterministic references
def find_references(chunk: Chunk, known_sections: set[str]) -> set[str]:
    """Section labels in the same document that this chunk cites ("Article 55", "section 8", "AC-2")."""
    text = chunk.text
    found: set[str] = set()
    if chunk.doc_id == "gdpr" or chunk.section.startswith(("Article", "Recital")):
        for m in re.finditer(r"\bArticles?\s+((?:\d+(?:\(\d+\))?(?:\s*(?:,|and|or)\s*)?)+)", text):
            found |= {f"Article {n}" for n in re.findall(r"(?<!\()\b(\d{1,3})\b(?!\))", m.group(1))}
    for m in re.finditer(r"\b[Ss]ections?\s+(\d{1,3})\b", text):
        found.add(f"Section {m.group(1)}")
    for m in re.finditer(r"\b([A-Z]{2}-\d{1,2})\b", text):
        found.add(m.group(1))
    for m in re.finditer(r"\b([A-Z]{2}\.[A-Z]{2})(?:-\d{2})?\b", text):
        found.add(m.group(1))
    found.discard(chunk.section)
    return found & known_sections


# ------------------------------------------------------------------------- chunk graph
@dataclass
class GNode:
    key: str
    label: str  # Regulation | Section | Obligation | Entity | Right | Penalty | Definition
    name: str
    description: str = ""
    doc_id: str = ""


@dataclass
class GEdge:
    source: str  # node key
    rel: str
    target: str


@dataclass
class ChunkGraph:
    chunk: Chunk
    nodes: list[GNode] = field(default_factory=list)
    edges: list[GEdge] = field(default_factory=list)


def build_chunk_graph(chunk: Chunk, extraction: Extraction | None, known_sections: set[str]) -> ChunkGraph:
    """Combine the deterministic structure with the (validated) LLM extraction for one chunk."""
    g = ChunkGraph(chunk)
    reg_key, sec_key = f"Regulation:{chunk.doc_id}", section_key(chunk.doc_id, chunk.section)
    g.nodes.append(GNode(reg_key, "Regulation", chunk.doc_title, doc_id=chunk.doc_id))
    g.nodes.append(GNode(sec_key, "Section", f"{chunk.doc_title} {chunk.section}", chunk.section_title,
                         chunk.doc_id))
    g.edges.append(GEdge(reg_key, "CONTAINS", sec_key))
    for ref in sorted(find_references(chunk, known_sections)):
        ref_key = section_key(chunk.doc_id, ref)
        g.nodes.append(GNode(ref_key, "Section", f"{chunk.doc_title} {ref}", "", chunk.doc_id))
        g.edges.append(GEdge(sec_key, "REFERENCES", ref_key))

    if extraction is None:
        return g
    by_name: dict[str, GNode] = {}
    for n in extraction.nodes:
        key = node_key(n.type, n.name, chunk.doc_id)
        if any(x.key == key for x in g.nodes):
            continue
        node = GNode(key, n.type, n.name.strip(), n.description.strip(), chunk.doc_id)
        g.nodes.append(node)
        by_name[normalize_name(n.name)] = node
        if n.type in SECTION_REL_FOR:
            g.edges.append(GEdge(sec_key, SECTION_REL_FOR[n.type], key))
    for r in extraction.relations:
        src, tgt = by_name.get(normalize_name(r.source)), by_name.get(normalize_name(r.target))
        if src and tgt and src.key != tgt.key and (src.label, r.relation, tgt.label) in ALLOWED_TRIPLES:
            g.edges.append(GEdge(src.key, r.relation, tgt.key))
    return g


# Known cross-regulation equivalents (domain knowledge). Embedding similarity finds the rest.
SEED_SIMILAR = [
    ("data fiduciary", "controller"),
    ("data principal", "data subject"),
    ("data processor", "processor"),
    ("data protection board", "supervisory authority"),
]
