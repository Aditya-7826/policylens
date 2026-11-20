"""Look and feel for the Streamlit UI: a dark, gold-accented "law office" theme.

Colours and fonts live in .streamlit/config.toml; this module adds what the theme
options can't express (nav bar, hero, cards) as CSS plus small HTML builders.
"""

from __future__ import annotations

from html import escape
from urllib.parse import quote

GOLD = "#D9AE5F"

# Known corpus documents: (jurisdiction, title, blurb, starter question)
DOCS = {
    "dpdp_act_2023": ("India", "DPDP Act 2023",
                      "Consent, notices, breach intimation, Data Fiduciary duties and the penalty Schedule.",
                      "What must a Data Fiduciary do after a personal data breach under the DPDP Act?"),
    "gdpr": ("European Union", "GDPR",
             "Lawful bases, data subject rights, controller and processor obligations, and fines.",
             "When must a controller notify a personal data breach under GDPR?"),
    "nist_csf_2.0": ("United States · NIST", "CSF 2.0",
                     "Govern, Identify, Protect, Detect, Respond and Recover: outcomes for cyber risk.",
                     "Which NIST CSF 2.0 category covers incident response planning?"),
    "nist_sp_800-53r5": ("United States · NIST", "SP 800-53 Rev. 5",
                         "The security and privacy controls catalogue, from access control to supply chain risk.",
                         "What does NIST SP 800-53 AC-2 require for account management?"),
}

STEPS = [
    ("Decompose", "Your question is classified and split into focused sub-questions, one per regulation it touches."),
    ("Retrieve", "Each sub-question searches three ways at once: meaning (vectors), exact wording (BM25) and the "
                 "knowledge graph of obligations, rights and penalties."),
    ("Verify", "The answer cites a source passage for every factual sentence. Each citation is checked, and "
               "sentences their source doesn't support are removed."),
    ("Cite or abstain", "You get a confidence score and clickable sources, or a plain \"not enough evidence\" "
                        "instead of a guess."),
]


def scales_uri(stroke: float, opacity: float = 1.0) -> str:
    """Balance-scales mark as a CSS url(): Streamlit's HTML sanitizer strips inline <svg>."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" fill="none" stroke="{GOLD}"
 stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}">
<circle cx="100" cy="24" r="7"/><line x1="100" y1="31" x2="100" y2="168"/>
<line x1="34" y1="52" x2="166" y2="52"/><circle cx="100" cy="52" r="5"/>
<line x1="40" y1="52" x2="20" y2="114"/><line x1="40" y1="52" x2="60" y2="114"/>
<line x1="160" y1="52" x2="140" y2="114"/><line x1="160" y1="52" x2="180" y2="114"/>
<path d="M14 114 Q40 140 66 114 Z"/><path d="M134 114 Q160 140 186 114 Z"/>
<path d="M70 176 H130 M80 168 H120"/></svg>"""
    return f'url("data:image/svg+xml,{quote(svg)}")'


CSS = f"""<style>
:root {{
  --gold: {GOLD}; --gold-soft: rgba(217,174,95,.14); --ink: #0A0A0A; --panel: #121110;
  --line: #2A2722; --text: #EDE8DF; --muted: #A59F94;
  --serif: 'Playfair Display', Georgia, serif;
}}
/* Raleway and Playfair default to old-style figures, which make "800-53" look uneven */
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp b, .stApp i {{
  font-variant-numeric: lining-nums; font-feature-settings: "lnum" 1; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"] {{ display: none; }}  /* it sat on top of the nav CTA */
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
section.stMain {{ scroll-behavior: smooth; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}

/* ---- nav ---- */
.pl-nav {{ display:flex; align-items:center; gap:2rem; padding:.6rem 0 1.4rem; border-bottom:1px solid var(--line); }}
.pl-brand {{ display:flex; align-items:center; gap:.8rem; text-decoration:none !important; color:var(--text) !important; }}
.pl-mark {{ width:44px; height:44px; display:grid; place-items:center; border:2px solid var(--gold);
  transform:rotate(45deg); flex:none; }}
.pl-mark::after {{ content:""; width:26px; height:26px; transform:rotate(-45deg);
  background:{scales_uri(12)} center/contain no-repeat; }}
.pl-brand b {{ font-family:var(--serif); font-size:1.45rem; letter-spacing:.01em; display:block; line-height:1.1; }}
.pl-brand small {{ color:var(--muted); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; }}
.pl-links {{ margin-left:auto; display:flex; gap:1.8rem; }}
.pl-links a {{ color:var(--muted) !important; text-decoration:none !important; font-size:.95rem; font-weight:500; }}
.pl-links a:hover, .pl-links a.on {{ color:var(--text) !important; }}
.pl-cta {{ background:var(--gold); color:var(--ink) !important; text-decoration:none !important; padding:.7rem 1.3rem;
  font-size:.8rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; white-space:nowrap; }}
.pl-cta:hover {{ filter:brightness(1.08); }}

/* ---- hero ---- */
.pl-hero {{ position:relative; overflow:hidden; display:grid; grid-template-columns:1.25fr 1fr; align-items:center;
  min-height:470px; margin:0 0 2.5rem; padding:3rem 3rem; border-bottom:1px solid var(--line);
  background: radial-gradient(ellipse at 78% 45%, rgba(217,174,95,.16), transparent 55%),
              linear-gradient(180deg, #0d0c0b, #0A0A0A); }}
.pl-eyebrow {{ font-family:var(--serif); color:var(--muted); font-size:1.15rem; margin:0 0 .8rem; }}
.pl-hero h1 {{ font-family:var(--serif); font-size:clamp(2.4rem, 5vw, 4.1rem); line-height:1.15; margin:0;
  color:var(--text); font-weight:700; padding:0; }}
.pl-rule {{ display:block; width:44px; height:2px; background:var(--gold); margin:1.8rem 0; }}
.pl-lede {{ color:var(--text); opacity:.85; max-width:34rem; line-height:1.75; margin:0 0 2rem; }}
.pl-btn-outline {{ display:inline-block; border:2px solid var(--gold); color:var(--gold) !important;
  text-decoration:none !important; padding:.75rem 1.5rem; font-size:.82rem; font-weight:600; letter-spacing:.1em;
  text-transform:uppercase; }}
.pl-btn-outline:hover {{ background:var(--gold-soft); }}
.pl-hero-art {{ height:360px; background:{scales_uri(1.6, .6)} center/contain no-repeat;
  filter:drop-shadow(0 0 28px rgba(217,174,95,.25)); }}

/* ---- stats strip ---- */
.pl-stats {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); margin:0 0 3.5rem; }}
.pl-stat {{ padding:1.6rem 1.4rem; border-right:1px solid var(--line); }}
.pl-stat:last-child {{ border-right:none; }}
.pl-stat b {{ font-family:var(--serif); font-size:2.3rem; color:var(--gold); display:block; line-height:1.1; }}
.pl-stat span {{ color:var(--muted); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
  display:block; margin-top:.4rem; line-height:1.5; }}

/* ---- section headings ---- */
.pl-section {{ scroll-margin-top:4rem; margin:1rem 0 1.6rem; text-align:center; }}
.pl-section p {{ color:var(--gold); font-size:.78rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase;
  margin:0 0 .5rem; }}
.pl-section h2 {{ font-family:var(--serif); font-size:clamp(1.8rem,3.4vw,2.6rem); margin:0; padding:0; color:var(--text); }}
.pl-section .pl-rule {{ margin:1.2rem auto 0; }}

/* ---- coverage cards (Streamlit containers keyed plcard-*) ---- */
[class*="st-key-plcard"] {{ background:var(--panel); border:1px solid var(--line); border-top:2px solid var(--gold);
  padding:1.5rem 1.4rem 1.2rem; height:100%; transition:border-color .2s; }}
[class*="st-key-plcard"]:hover {{ border-color:var(--gold); }}
.pl-card-tag {{ color:var(--gold); font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
.pl-card h3 {{ font-family:var(--serif); font-size:1.45rem; margin:.4rem 0 .6rem; padding:0; color:var(--text); }}
.pl-card p {{ color:var(--muted); font-size:.92rem; line-height:1.6; min-height:4.4rem; margin:0 0 .9rem; }}
.pl-card .pl-count {{ color:var(--text); font-size:.82rem; border-top:1px solid var(--line); padding-top:.7rem; }}
.pl-card .pl-count b {{ color:var(--gold); }}

/* ---- method steps ---- */
.pl-steps {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0; border-top:1px solid var(--line);
  margin:0 0 3.5rem; }}
.pl-step {{ padding:1.6rem 1.4rem 0 0; }}
.pl-step i {{ font-family:var(--serif); font-style:normal; color:var(--gold); font-size:2.6rem; opacity:.9; }}
.pl-step h4 {{ font-family:var(--serif); font-size:1.2rem; margin:.3rem 0 .5rem; padding:0; color:var(--text); }}
.pl-step p {{ color:var(--muted); font-size:.9rem; line-height:1.65; margin:0; }}

/* ---- buttons ---- */
.stButton button {{ text-transform:uppercase; letter-spacing:.08em; font-size:.78rem; font-weight:600;
  border:1px solid var(--gold); color:var(--gold); background:transparent; }}
.stButton button:hover {{ background:var(--gold-soft); color:var(--gold); border-color:var(--gold); }}
.stButton button[kind="primary"] {{ background:var(--gold); color:var(--ink); }}
.stButton button[kind="primary"]:hover {{ filter:brightness(1.08); color:var(--ink); }}
.stButton button p {{ font-size:.78rem; }}
/* example questions read as quotes, not shouty buttons */
[class*="st-key-plq"] button {{ text-transform:none; letter-spacing:0; text-align:left; justify-content:flex-start;
  border:1px solid var(--line); border-left:2px solid var(--gold); color:var(--text); background:var(--panel);
  padding:1rem 1.2rem; min-height:4.5rem; }}
[class*="st-key-plq"] button p {{ font-family:var(--serif); font-size:1.02rem; text-align:left; }}
[class*="st-key-plq"] button:hover {{ border-color:var(--gold); color:var(--text); }}

/* ---- chat ---- */
[data-testid="stChatMessage"] {{ background:var(--panel); border:1px solid var(--line); border-radius:0;
  padding:1.1rem 1.3rem; }}
[data-testid="stChatMessageAvatarCustom"], [data-testid^="stChatMessageAvatar"] {{ background:var(--gold);
  color:var(--ink); border-radius:0; }}
[data-testid="stChatInput"] {{ border:1px solid var(--line); border-radius:0; }}
[data-testid="stChatInput"]:focus-within {{ border-color:var(--gold); }}
[data-testid="stExpander"] details {{ border-color:var(--line); border-radius:0; }}
[data-testid="stPopover"] button {{ text-transform:none; letter-spacing:0; border-color:var(--line); color:var(--text); }}
[data-testid="stPopover"] button:hover {{ border-color:var(--gold); }}

.pl-footer {{ border-top:1px solid var(--line); margin-top:3rem; padding:1.4rem 0 .5rem; color:var(--muted);
  font-size:.82rem; display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}

@media (max-width: 900px) {{
  .pl-links {{ display:none; }}
  .pl-cta {{ margin-left:auto; padding:.6rem .9rem; font-size:.7rem; }}
  .pl-brand small {{ display:none; }}
  .pl-hero {{ grid-template-columns:1fr; padding:2rem 1.2rem; min-height:0; }}
  .pl-hero-art {{ display:none; }}
  .pl-stats, .pl-steps {{ grid-template-columns:repeat(2,1fr); }}
  .pl-stat:nth-child(2) {{ border-right:none; }}
  .pl-stat:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }}
}}
</style>"""


def nav(chatting: bool) -> str:
    links = "" if chatting else (
        '<div class="pl-links"><a class="on" href="#pl-home">Home</a><a href="#pl-coverage">Coverage</a>'
        '<a href="#pl-method">How it works</a><a href="#pl-ask">Ask</a></div>')
    cta = '<a class="pl-cta" href="#pl-ask">Ask a question</a>' if not chatting else ""
    return f"""<nav class="pl-nav" id="pl-home">
<a class="pl-brand" href="#pl-home"><span class="pl-mark"></span>
<span><b>PolicyLens</b><small>Compliance, cited</small></span></a>{links}{cta}</nav>"""


def hero() -> str:
    return """<section class="pl-hero">
<div><p class="pl-eyebrow">Welcome to PolicyLens</p>
<h1>Compliance Answers<br>You Can Defend</h1><span class="pl-rule"></span>
<p class="pl-lede">Ask about India's DPDP Act, the GDPR, NIST CSF 2.0 or SP 800-53. Every sentence cites the
clause it came from, every citation is checked, and when the evidence is thin PolicyLens says so instead of guessing.</p>
<a class="pl-btn-outline" href="#pl-method">How it works</a></div>
<div class="pl-hero-art"></div></section>"""


def stats_strip(stats: dict) -> str:
    docs = stats.get("documents", {}) if "error" not in stats else {}
    nodes = (stats.get("graph") or {}).get("nodes", {})
    items = [
        (len(docs) or len(DOCS), "Documents indexed"),
        (sum(d["chunks"] for d in docs.values()), "Indexed passages"),
        (nodes.get("Obligation", 0), "Obligations mapped"),
        (nodes.get("Right", 0) + nodes.get("Penalty", 0), "Rights & penalties"),
    ]
    cells = "".join(f'<div class="pl-stat"><b>{n:,}</b><span>{label}</span></div>' for n, label in items)
    return f'<div class="pl-stats">{cells}</div>'


def section_head(anchor: str, eyebrow: str, title: str) -> str:
    return (f'<div class="pl-section" id="{anchor}"><p>{eyebrow}</p><h2>{title}</h2>'
            f'<span class="pl-rule"></span></div>')


def doc_card(doc_id: str, chunks: int) -> tuple[str, str]:
    """HTML body for a coverage card, and the starter question for its button."""
    tag, title, blurb, question = DOCS.get(doc_id, (
        "Your upload", doc_id.replace("_", " ").title(), "A document you added. Ask about anything it covers.",
        f"Summarize the key obligations in {doc_id.replace('_', ' ')}."))
    html = (f'<div class="pl-card"><span class="pl-card-tag">{escape(tag)}</span><h3>{escape(title)}</h3>'
            f'<p>{escape(blurb)}</p><div class="pl-count"><b>{chunks:,}</b> indexed passages</div></div>')
    return html, question


def steps() -> str:
    cells = "".join(f'<div class="pl-step"><i>{i:02d}</i><h4>{t}</h4><p>{escape(d)}</p></div>'
                    for i, (t, d) in enumerate(STEPS, 1))
    return f'<div class="pl-steps">{cells}</div>'


FOOTER = ('<div class="pl-footer"><span>PolicyLens answers only from the documents it has indexed.</span>'
          '<span>Research aid, not legal advice.</span></div>')
