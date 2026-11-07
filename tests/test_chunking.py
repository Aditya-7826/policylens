from pathlib import Path

from ingest.chunk import chunk_document, n_tokens, split_units, window
from ingest.parse import Document, Page, _group_margin_notes, html_to_text, strip_running_headers


def doc(doc_id, text, kind="txt"):
    return Document(doc_id, doc_id.upper(), Path(f"{doc_id}.txt"), [Page(1, text)], kind)


GDPR_LIKE = """Whereas:
(1)
The protection of natural persons is a fundamental right.
(2)
The principles should respect fundamental rights.
Article 1
Subject-matter and objectives
1. This Regulation lays down rules.
Article 33
Notification of a personal data breach to the supervisory authority
1. In the case of a personal data breach, the controller shall notify within 72 hours.
(1)
footnote-like line after the articles stays inside Article 33.
"""


def test_gdpr_articles_and_recitals():
    units = split_units(doc("gdpr", GDPR_LIKE))
    labels = [u.section for u in units]
    assert labels == ["Preamble", "Recital 1", "Recital 2", "Article 1", "Article 33"]
    art33 = units[-1]
    assert art33.title.startswith("Notification of a personal data breach")
    assert any("footnote-like" in ln for ln in art33.lines)  # recital rule is off after Article 1


def test_chunk_ids_are_stable_and_readable():
    a = chunk_document(doc("gdpr", GDPR_LIKE))
    b = chunk_document(doc("gdpr", GDPR_LIKE))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert "gdpr::article-33::0" in {c.chunk_id for c in a}
    c = next(c for c in a if c.section == "Article 33")
    assert c.citation() == "[GDPR, Article 33, gdpr::article-33::0]"
    assert c.context_text.startswith("GDPR - Article 33: Notification")


def test_gazette_marker_lines():
    text = "Section 8: General obligations of Data Fiduciary.\n8. (1) A Data Fiduciary shall comply.\n" \
           "Section 9: Processing of personal data of children.\n9. (1) The Data Fiduciary shall..."
    units = split_units(doc("some_act", text, kind="gazette"))
    assert [(u.section, u.title) for u in units] == [
        ("Section 8", "General obligations of Data Fiduciary."),
        ("Section 9", "Processing of personal data of children."),
    ]


def test_csf_named_groups_keep_description():
    text = "GOVERN (GV): Strategy is established\n• Organizational Context (GV.OC): The circumstances are understood\n" \
           "o GV.OC-01: The mission is understood"
    units = split_units(doc("nist_csf_2.0", text, kind="pdf"))
    assert [(u.section, u.title) for u in units] == [("GV", "Govern"), ("GV.OC", "Organizational Context")]
    assert units[1].lines[0].startswith("• Organizational Context (GV.OC): The circumstances")


def test_windows_respect_max_tokens_with_overlap():
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(30)]
    wins = window(paras, max_tokens=200, overlap=70)
    assert len(wins) > 5
    assert all(n_tokens(w) <= 200 for w in wins)
    # consecutive windows share an overlapping paragraph
    assert wins[0].split("\n")[-1] == wins[1].split("\n")[0]


def test_very_long_paragraph_is_hard_split():
    wins = window(["x" * 20000], max_tokens=300, overlap=50)
    assert all(n_tokens(w) <= 300 for w in wins)


def test_strip_running_headers():
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    pages = [Page(i, f"NIST SP 800-53, REV. 5\nCHAPTER THREE PAGE {i}\nBody about {words[i - 1]}.\n{i}")
             for i in range(1, 11)]
    out = strip_running_headers(pages)
    assert [p.text for p in out] == [f"Body about {w}." for w in words]


def test_margin_notes_grouping_skips_act_citations():
    def w(text, top, x0=486):
        return {"text": text, "top": top, "x0": x0}
    words = [w("Short", 100), w("title", 100, 510), w("and", 100, 530), w("commencement.", 110),
             w("24", 300), w("of", 300, 500), w("1997.", 300, 510),
             w("Notice.", 400)]
    notes = _group_margin_notes(words)
    assert [n[1] for n in notes] == ["Short title and commencement.", "Notice."]


def test_html_blocks_become_lines():
    html = "<html><body><p class='ti-art'>Article 5</p><p>Principles</p><script>x()</script></body></html>"
    assert html_to_text(html) == "Article 5\nPrinciples"


def test_definition_units_get_small_windows():
    defs = "\n".join(f"({chr(97 + i)}) “term{i}” means " + "a defined thing in this Act " * 8 + ";" for i in range(20))
    text = "Section 2: Definitions.\n2. In this Act,\n" + defs + "\nSection 3: Application of Act.\n3. " + "word " * 300
    chunks = chunk_document(doc("act", text, kind="gazette"), max_tokens=700)
    s2 = [c for c in chunks if c.section == "Section 2"]
    s3 = [c for c in chunks if c.section == "Section 3"]
    assert len(s2) > 3 and all(c.n_tokens <= 250 for c in s2)
    assert len(s3) == 1  # normal sections keep the normal window


def test_kg_docs_keeps_dotted_ids(tmp_path):
    import pytest

    from ingest.pipeline import parse_kg_docs

    for name in ("nist_csf_2.0.pdf", "gdpr.html"):
        (tmp_path / name).write_bytes(b"x")
    assert parse_kg_docs("nist_csf_2.0, gdpr.html", tmp_path) == ["nist_csf_2.0", "gdpr"]
    with pytest.raises(SystemExit):
        parse_kg_docs("nist_csf_2", tmp_path)
