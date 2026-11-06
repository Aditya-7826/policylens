"""Neo4j graph store + the graph step of ingestion.

Every knowledge node has the label :Node plus its type label (:Entity, :Obligation, ...),
and a unique `key` (see ingest.kg.node_key). Relationships carry `chunks`: the chunk IDs
that produced them, so a changed chunk can be "forgotten" and re-extracted cleanly.
"""

from __future__ import annotations

import logging
import re

from core.config import get_settings
from ingest.chunk import Chunk
from ingest.kg import (
    SEED_SIMILAR, ChunkGraph, Extraction, build_chunk_graph, extraction_messages, normalize_name,
    worth_extracting,
)
from ingest.store import ChunkStore

log = logging.getLogger(__name__)

LABELS = {"Regulation", "Section", "Obligation", "Entity", "Right", "Penalty", "Definition"}
RELS = {"CONTAINS", "REQUIRES", "APPLIES_TO", "DEFINES", "PENALIZES", "REFERENCES", "SIMILAR_TO"}
KNOWLEDGE_LABELS = ["Obligation", "Entity", "Right", "Penalty", "Definition"]
TRAVERSE = "|".join(sorted(RELS))


class GraphStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        s = get_settings()
        # notifications off: Neo4j warns about relationship types that don't exist yet (e.g. before
        # SIMILAR_TO links are built), which is expected and just noise in the logs.
        self.driver = GraphDatabase.driver(uri or s.neo4j_uri, auth=(user or s.neo4j_user,
                                                                     password or s.neo4j_password),
                                           notifications_min_severity="OFF")

    def close(self):
        self.driver.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.run("CREATE CONSTRAINT node_key IF NOT EXISTS FOR (n:Node) REQUIRE n.key IS UNIQUE")
        self.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self.run("CREATE FULLTEXT INDEX node_text IF NOT EXISTS FOR (n:Node) ON EACH [n.name, n.description]")

    # ------------------------------------------------------------------ writes
    def forget_chunk(self, chunk_id: str):
        """Remove everything a chunk contributed (before re-extracting it, or when it is deleted)."""
        self.run("""MATCH ()-[r]->() WHERE $cid IN r.chunks
                    SET r.chunks = [x IN r.chunks WHERE x <> $cid]
                    WITH r WHERE size(r.chunks) = 0 DELETE r""", cid=chunk_id)
        self.run("MATCH (c:Chunk {id: $cid}) DETACH DELETE c", cid=chunk_id)

    def prune_orphans(self) -> int:
        rows = self.run("""MATCH (n:Node) WHERE NOT (n)-[:MENTIONED_IN]->(:Chunk)
                           AND NOT n:Regulation DETACH DELETE n RETURN count(*) AS n""")
        return rows[0]["n"] if rows else 0

    def write(self, cg: ChunkGraph):
        c = cg.chunk
        with self.driver.session() as s:
            s.execute_write(self._write_tx, c, cg)

    @staticmethod
    def _write_tx(tx, c: Chunk, cg: ChunkGraph):
        tx.run("""MERGE (ch:Chunk {id: $id})
                  SET ch.doc_id = $doc_id, ch.section = $section, ch.title = $title""",
               id=c.chunk_id, doc_id=c.doc_id, section=c.section, title=c.section_title)
        for label in {n.label for n in cg.nodes}:
            assert label in LABELS
            rows = [{"key": n.key, "name": n.name, "desc": n.description, "doc": n.doc_id}
                    for n in cg.nodes if n.label == label]
            tx.run(f"""UNWIND $rows AS row
                       MERGE (n:Node:{label} {{key: row.key}})
                       ON CREATE SET n.name = row.name, n.description = row.desc, n.doc_ids = [row.doc]
                       ON MATCH SET n.doc_ids = CASE WHEN row.doc IN n.doc_ids THEN n.doc_ids
                                                     ELSE n.doc_ids + row.doc END,
                                    n.description = CASE WHEN size(coalesce(n.description, '')) = 0
                                                         THEN row.desc ELSE n.description END
                       WITH n
                       MATCH (ch:Chunk {{id: $cid}})
                       MERGE (n)-[:MENTIONED_IN]->(ch)""", rows=rows, cid=c.chunk_id)
        for rel in {e.rel for e in cg.edges}:
            assert rel in RELS
            rows = [{"s": e.source, "t": e.target} for e in cg.edges if e.rel == rel]
            tx.run(f"""UNWIND $rows AS row
                       MATCH (a:Node {{key: row.s}}), (b:Node {{key: row.t}})
                       MERGE (a)-[r:{rel}]->(b)
                       ON CREATE SET r.chunks = [$cid]
                       ON MATCH SET r.chunks = CASE WHEN $cid IN r.chunks THEN r.chunks
                                                    ELSE r.chunks + $cid END""", rows=rows, cid=c.chunk_id)

    def link_similar(self, threshold: float = 0.86, progress=print) -> int:
        """SIMILAR_TO between same-type nodes from *different* regulations, by embedding similarity
        of "name: description", plus the curated seed pairs."""
        from core.models import get_embedder

        self.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
        created = 0
        for label in KNOWLEDGE_LABELS:
            nodes = self.run(f"MATCH (n:{label}) RETURN n.key AS key, n.name AS name, "
                             "coalesce(n.description, '') AS d, n.doc_ids AS docs")
            if len(nodes) < 2:
                continue
            vecs = get_embedder().embed_documents([f"{n['name']}: {n['d']}" for n in nodes])
            import numpy as np

            m = np.array(vecs)
            sims = m @ m.T
            pairs = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if sims[i, j] >= threshold and not set(nodes[i]["docs"]) & set(nodes[j]["docs"]):
                        pairs.append({"a": nodes[i]["key"], "b": nodes[j]["key"], "score": float(sims[i, j])})
            if pairs:
                self.run("""UNWIND $pairs AS p MATCH (a:Node {key: p.a}), (b:Node {key: p.b})
                            MERGE (a)-[r:SIMILAR_TO]->(b) SET r.score = p.score, r.chunks = ['similarity']""",
                         pairs=pairs)
            created += len(pairs)
        seeds = [{"a": f"Entity:{normalize_name(a)}", "b": f"Entity:{normalize_name(b)}"} for a, b in SEED_SIMILAR]
        rows = self.run("""UNWIND $pairs AS p MATCH (a:Node {key: p.a}), (b:Node {key: p.b})
                           MERGE (a)-[r:SIMILAR_TO]->(b) SET r.score = 1.0, r.chunks = ['seed']
                           RETURN count(*) AS n""", pairs=seeds)
        created += rows[0]["n"] if rows else 0
        progress(f"[graph] {created} SIMILAR_TO links")
        return created

    # ------------------------------------------------------------------ reads (used by rag/)
    def search_nodes(self, text: str, k: int = 8) -> list[dict]:
        q = lucene_query(text)
        if not q:
            return []
        return self.run("""CALL db.index.fulltext.queryNodes('node_text', $q) YIELD node, score
                           WHERE NOT node:Regulation
                           RETURN node.key AS key, node.name AS name, labels(node) AS labels, score
                           LIMIT $k""", q=q, k=k)

    def expand(self, seed_keys: list[str], hops: int = 2, limit: int = 300) -> list[dict]:
        """Walk 0..hops relationships from the seeds (never through a Regulation hub) and
        return the chunks mentioning each reached node, with the path that got there."""
        hops = max(0, min(int(hops), 3))
        return self.run(f"""
            MATCH (seed:Node) WHERE seed.key IN $keys
            MATCH p = (seed)-[:{TRAVERSE}*0..{hops}]-(n:Node)
            WHERE none(x IN nodes(p) WHERE x:Regulation)
            MATCH (n)-[:MENTIONED_IN]->(c:Chunk)
            WITH seed, n, c, p ORDER BY length(p)
            WITH seed.key AS seed, c.id AS chunk_id, head(collect(p)) AS p
            RETURN seed, chunk_id, length(p) AS hops,
                   [x IN nodes(p) | {{name: x.name, label: [l IN labels(x) WHERE l <> 'Node'][0]}}] AS path_nodes,
                   [r IN relationships(p) | type(r)] AS path_rels
            LIMIT $limit""", keys=seed_keys, limit=limit)

    def subgraph(self, chunk_ids: list[str], max_nodes: int = 40) -> dict:
        """Nodes mentioned in the given chunks and the edges among them (for the UI graph view)."""
        nodes = self.run("""MATCH (n:Node)-[:MENTIONED_IN]->(c:Chunk) WHERE c.id IN $ids AND NOT n:Regulation
                            RETURN DISTINCT n.key AS key, n.name AS name,
                                   [l IN labels(n) WHERE l <> 'Node'][0] AS label LIMIT $m""",
                         ids=chunk_ids, m=max_nodes)
        keys = [n["key"] for n in nodes]
        edges = self.run("""MATCH (a:Node)-[r]->(b:Node) WHERE a.key IN $keys AND b.key IN $keys
                            AND type(r) <> 'MENTIONED_IN'
                            RETURN a.key AS source, type(r) AS rel, b.key AS target""", keys=keys)
        return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        labels = self.run("MATCH (n:Node) UNWIND [l IN labels(n) WHERE l <> 'Node'] AS l "
                          "RETURN l AS label, count(*) AS n ORDER BY n DESC")
        rels = self.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC")
        return {"nodes": {r["label"]: r["n"] for r in labels}, "relationships": {r["rel"]: r["n"] for r in rels}}


_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_STOP = set("a an and are as at be by can do does for from has have how i in is it its of on or "
            "should that the their there these this to was what when where which who why will with "
            "under after before about must company compare comparison between versus vs "
            # regulation names: every Section node's name contains them, so they match everything
            "act dpdp gdpr india indias indian eu european regulation nist csf framework sp 800-53 rev "
            "digital personal protection 2023 2016 general".split())


def lucene_query(text: str) -> str:
    """Plain keywords OR-ed together, with Lucene special characters escaped."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", text)
             if re.sub(r"'s$", "", w.lower()) not in _STOP and len(w) > 2]
    return " OR ".join(_LUCENE_SPECIAL.sub(r"\\\1", w) for w in words[:20])


# ---------------------------------------------------------------------- ingestion step
def build_graph(store: ChunkStore, doc_ids: list[str] | None = None, limit: int | None = None,
                removed_ids: list[str] | None = None, progress=print, graph: GraphStore | None = None,
                llm=None) -> dict:
    """Extract the graph for every chunk not yet processed. Stops cleanly when the LLM
    budget runs out; the next run continues where this one stopped."""
    from core.llm import LLMUnavailable, get_llm

    graph = graph or GraphStore()
    if not graph.ping():
        progress("[graph] Neo4j is not reachable - skipped. Start it with: docker compose up -d neo4j")
        return {"status": "neo4j unavailable"}
    llm = llm or get_llm()
    graph.setup()
    for cid in removed_ids or []:
        graph.forget_chunk(cid)

    all_chunks = store.all()
    sections_by_doc: dict[str, set[str]] = {}
    for c in all_chunks:
        sections_by_doc.setdefault(c.doc_id, set()).add(c.section)

    pending = store.pending_kg(doc_ids)
    counts = {"done": 0, "skipped": 0, "error": 0, "structure_only": 0}
    llm_calls = 0
    llm_ok = True  # flips to False when the budget/limit is hit; the rest gets structure only
    progress(f"[graph] {len(pending)} chunks pending extraction")
    for i, chunk in enumerate(pending):
        ok, _reason = worth_extracting(chunk)
        extraction = None
        if ok and llm_ok and limit is not None and llm_calls >= limit:
            llm_ok = False
            progress(f"[graph] --kg-limit {limit} reached; remaining chunks get structure only for now")
        if ok and llm_ok:
            try:
                extraction = llm.chat_json(extraction_messages(chunk), Extraction, size="small", max_tokens=900)
                llm_calls += 1
            except LLMUnavailable as exc:
                llm_ok = False
                progress(f"[graph] LLM unavailable ({exc}); remaining chunks get structure only for now")
            except Exception as exc:  # invalid JSON twice etc. - record and move on
                log.warning("extraction failed for %s: %s", chunk.chunk_id, exc)
                store.mark_kg(chunk, "error")
                counts["error"] += 1
                continue
        graph.forget_chunk(chunk.chunk_id)
        graph.write(build_chunk_graph(chunk, extraction, sections_by_doc.get(chunk.doc_id, set())))
        if ok and extraction is None:
            counts["structure_only"] += 1  # stays pending: the next run extracts it
        else:
            status = "done" if ok else "skipped"
            store.mark_kg(chunk, status)
            counts[status] += 1
        if (i + 1) % 25 == 0:
            progress(f"[graph] {i + 1}/{len(pending)} (done {counts['done']}, skipped {counts['skipped']}, "
                     f"structure only {counts['structure_only']})")

    if counts["done"] or removed_ids:
        counts["orphans_pruned"] = graph.prune_orphans()
        counts["similar_links"] = graph.link_similar(progress=progress)
    counts["graph"] = graph.stats()
    return counts
