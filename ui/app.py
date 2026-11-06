"""Streamlit chat UI for PolicyLens.

    uv run streamlit run ui/app.py

Talks to the FastAPI service (API_URL, default http://localhost:8000) over SSE.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import streamlit as st
from httpx_sse import connect_sse

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolicyLens", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "Under India's DPDP Act, what must a company do after a personal data breach, and how does that compare with GDPR?",
    "What is the maximum penalty under the DPDP Act for failing to take reasonable security safeguards?",
    "Which NIST CSF 2.0 category covers incident response planning?",
    "What does NIST SP 800-53 AC-2 require for account management?",
]
CONFIG_HELP = {
    "full": "Hybrid GraphRAG: decomposition + vector + BM25 + graph, reranking, verified citations, abstention",
    "hybrid": "Vector + BM25 fused with RRF (no graph, reranker or verification)",
    "vector": "Plain vector-search RAG baseline",
}
STATUS_ICON = {"supported": "✅", "partial": "🟡", "unsupported": "❌", "uncited": "⚠️", "connective": "·"}
LABEL_COLOR = {"Section": "#8892b0", "Obligation": "#e76f51", "Entity": "#2a9d8f", "Right": "#457b9d",
               "Penalty": "#d62828", "Definition": "#e9c46a", "Regulation": "#6c757d"}


# ----------------------------------------------------------------------------- API helpers
@st.cache_data(ttl=20, show_spinner=False)
def api_get(path: str, params: dict | None = None):
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}


def ask_stream(question: str, config: str, status):
    result = None
    with httpx.Client(timeout=httpx.Timeout(300, connect=10)) as client:
        with connect_sse(client, "POST", f"{API}/ask", json={"question": question, "config": config}) as es:
            for ev in es.iter_sse():
                data = json.loads(ev.data) if ev.data else {}
                if ev.event == "stage":
                    extra = ""
                    if data.get("sub_questions") and len(data["sub_questions"]) > 1:
                        extra = f" → {len(data['sub_questions'])} sub-questions ({data.get('query_type')})"
                    if "evidence" in data:
                        extra = f" → {data['evidence']} evidence chunks"
                    status.write(f"✓ {data['label']}{extra}")
                elif ev.event == "result":
                    result = data
                elif ev.event == "error":
                    raise RuntimeError(data.get("message", "unknown error"))
    return result


# ------------------------------------------------------------------------------- rendering
def confidence_badge(r: dict) -> str:
    if r["abstained"]:
        return ":orange-background[**Abstained — insufficient evidence**]"
    c = r.get("confidence")
    if c is None:
        return ":gray-background[confidence n/a (baseline mode)]"
    color = "green" if c >= 0.7 else "orange" if c >= 0.5 else "red"
    return f":{color}-background[**Confidence {c:.0%}**]"


def numbered_answer(r: dict) -> str:
    """Replace each full [doc, section, chunk_id] citation with a short [n] marker."""
    numbers = {c["chunk_id"]: i + 1 for i, c in enumerate(r["citations"])}

    def repl(m):
        cid = m.group(1).split(",")[-1].strip()
        return f" **[{numbers[cid]}]**" if cid in numbers else m.group(0)
    return re.sub(r"\s*\[([^\[\]]+::[^\[\]]+)\]", repl, r["answer"])


def show_sources(citations: list[dict], title="Sources"):
    if not citations:
        return
    st.caption(title)
    cols = st.columns(min(4, len(citations)))
    for i, c in enumerate(citations):
        with cols[i % len(cols)]:
            with st.popover(f"[{i + 1}] {c['doc_title']} {c['section']}", width="stretch"):
                st.markdown(f"**{c['doc_title']} — {c['section']}**"
                            + (f": {c['section_title']}" if c["section_title"] else ""))
                meta = [f"`{c['chunk_id']}`"]
                if c.get("page"):
                    meta.append(f"page {c['page']}")
                if c.get("rerank") is not None:
                    meta.append(f"rerank {c['rerank']:.2f}")
                if c.get("sources"):
                    meta.append("found by " + ", ".join(c["sources"]))
                st.caption(" · ".join(meta))
                st.text(c["text"])


def graph_html(sub: dict) -> str | None:
    if not sub.get("nodes"):
        return None
    from pyvis.network import Network

    net = Network(height="420px", width="100%", directed=True, cdn_resources="remote", bgcolor="#ffffff")
    for n in sub["nodes"]:
        net.add_node(n["key"], label=n["name"][:40], title=f"{n['label']}: {n['name']}",
                     color=LABEL_COLOR.get(n["label"], "#999"), shape="dot", size=14 if n["label"] != "Section" else 10)
    keys = {n["key"] for n in sub["nodes"]}
    for e in sub["edges"]:
        if e["source"] in keys and e["target"] in keys:
            net.add_edge(e["source"], e["target"], label=e["rel"], font={"size": 9}, arrows="to")
    net.repulsion(node_distance=140, spring_length=120)
    return net.generate_html()


def show_trace(r: dict):
    with st.expander("🔎 How I found this"):
        st.markdown(f"**Query type:** `{r['query_type']}` · **config:** `{r['config']}`")
        for i, sq in enumerate(r["sub_questions"], 1):
            regs = ", ".join(sq["regulations"]) or "all documents"
            st.markdown(f"{i}. {sq['question']}  \n   <small>searched: {regs}</small>", unsafe_allow_html=True)

        tabs = st.tabs(["Retrieval", "Graph paths", "Verification", "Timings"])
        with tabs[0]:
            for tr in r["retrieval"]:
                st.markdown(f"**{tr['question']}** — retrieved: " +
                            ", ".join(f"{k} {v}" for k, v in tr["counts"].items()))
                rows = [{"chunk": t["chunk_id"], **{f"{k} rank": v for k, v in t["ranks"].items()},
                         "RRF": t["rrf"], "rerank": t["rerank"]} for t in tr["top"]]
                st.dataframe(rows, width="stretch", hide_index=True)
        with tabs[1]:
            any_path = False
            for tr in r["retrieval"]:
                g = tr.get("graph")
                if isinstance(g, dict) and g.get("seeds"):
                    st.markdown("**Matched graph nodes:** " + ", ".join(
                        f"{s['name']} ({[x for x in s['label'] if x != 'Node'][0]})" for s in g["seeds"]))
                elif isinstance(g, str):
                    st.caption(f"graph: {g}")
                for t in tr["top"]:
                    p = t.get("graph_path")
                    if p and p.get("nodes"):
                        any_path = True
                        hops = []
                        for j, n in enumerate(p["nodes"]):
                            hops.append(f"`{n['name']}`")
                            if j < len(p["rels"]):
                                hops.append(f"—{p['rels'][j]}→")
                        st.markdown(f"- {' '.join(hops)} ⇒ `{t['chunk_id']}`")
            if not any_path:
                st.caption("No graph paths for this answer.")
        with tabs[2]:
            if r["sentences"]:
                st.dataframe([{"": STATUS_ICON.get(s["status"], ""), "sentence": s["text"], "status": s["status"],
                               "cites": ", ".join(f"E{h}" for h in s["handles"]), "similarity": s["similarity"],
                               "judge": s["judge"]} for s in r["sentences"]],
                             width="stretch", hide_index=True)
            if r["flagged"]:
                st.warning(f"{len(r['flagged'])} sentence(s) removed because their citations did not support them.")
            for n in r.get("notes", []):
                st.caption(f"note: {n}")
        with tabs[3]:
            st.json(r["timings"])

    if r["citations"]:
        with st.expander("🕸️ Entities used (knowledge graph)"):
            sub = api_get("/graph", {"chunk_ids": ",".join(c["chunk_id"] for c in r["citations"])})
            html = graph_html(sub) if "error" not in sub else None
            if html:
                st.components.v1.html(html, height=440)
            else:
                st.caption(sub.get("note") or sub.get("error") or "No graph entities for these chunks yet "
                           "(run ingestion with a GROQ_API_KEY to extract them).")


def render_result(r: dict):
    st.markdown(confidence_badge(r))
    if r["abstained"]:
        st.markdown(r["answer"])
        show_sources(r["evidence"][:6], "What I found (not enough to answer)")
    else:
        st.markdown(numbered_answer(r))
        show_sources(r["citations"])
    show_trace(r)


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚖️ PolicyLens")
    st.caption("Hybrid GraphRAG over DPDP, GDPR, NIST CSF 2.0 and SP 800-53")
    config = st.radio("Pipeline", list(CONFIG_HELP), format_func=lambda k: k, help="\n\n".join(
        f"**{k}**: {v}" for k, v in CONFIG_HELP.items()))
    st.caption(CONFIG_HELP[config])

    h = api_get("/health")
    if "error" in h:
        st.error(f"API not reachable at {API}. Start it with:\n\n`uv run uvicorn api.main:app`")
    else:
        st.markdown(f"**Chunks:** {h['chunks']} · **Neo4j:** {'🟢' if h['neo4j'] else '🔴'} · "
                    f"**LLM:** {', '.join(h['llm_providers']) or '🔴 no key'}")
        with st.expander("Stats"):
            stats = api_get("/stats")
            if "error" not in stats:
                st.dataframe([{"doc": d, **v} for d, v in stats["documents"].items()], hide_index=True)
                if stats.get("graph"):
                    st.json(stats["graph"])
                st.json(stats["llm"])

    with st.expander("Add documents"):
        files = st.file_uploader("PDF / HTML / TXT", type=["pdf", "html", "htm", "txt"], accept_multiple_files=True)
        with_graph = st.checkbox("Extract knowledge graph (uses LLM tokens)", value=True)
        if st.button("Ingest", disabled=not files):
            r = httpx.post(f"{API}/ingest", files=[("files", (f.name, f.getvalue())) for f in files],
                           data={"graph": str(with_graph).lower()}, timeout=120)
            if r.status_code == 200:
                st.session_state.ingest_job = r.json()["job_id"]
            else:
                st.error(r.text)
        if job := st.session_state.get("ingest_job"):
            j = api_get("/ingest/status", {"job_id": job})
            st.caption(f"job {job}: {j.get('status')}")
            st.code("\n".join(j.get("log", [])[-8:]) or "...")
            if st.button("Refresh status"):
                api_get.clear()
                st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------------- chat
st.session_state.setdefault("messages", [])
if not st.session_state.messages:
    st.markdown("#### Ask a compliance question")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_result(m["content"])

question = st.chat_input("e.g. What must a Data Fiduciary do after a breach?") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=True) as status:
            try:
                result = ask_stream(question, config, status)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"{exc}")
                result = None
        if result:
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
