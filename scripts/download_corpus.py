"""Download the public corpus into data/raw/ from official sources.

    uv run python -m scripts.download_corpus

Each document has a list of candidate URLs (official sites move files around).
Downloads are validated (PDF magic bytes / expected text) before being saved.
Anything that fails is printed with its URLs so you can download it by hand.
"""

import sys
from dataclasses import dataclass

import httpx

from core.config import get_settings


@dataclass
class Doc:
    filename: str
    title: str
    urls: list[str]
    must_contain: str  # sanity check on the downloaded bytes (case-insensitive)
    accept: str | None = None  # some APIs pick the format by Accept header


CORPUS = [
    Doc(
        "dpdp_act_2023.pdf",
        "India - Digital Personal Data Protection Act, 2023",
        [
            "https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
            "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf",
            "https://prsindia.org/files/bills_acts/acts_parliament/2023/Digital_Personal_Data_Protection_Act,_2023.pdf",
        ],
        "%PDF",
    ),
    Doc(
        "gdpr.html",
        "EU - General Data Protection Regulation (EU) 2016/679",
        [
            # EU Publications Office "Cellar" API: official text, no bot check (EUR-Lex often has one)
            "http://publications.europa.eu/resource/celex/32016R0679",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        ],
        "Article 33",
        accept="application/xhtml+xml",
    ),
    Doc(
        "nist_csf_2.0.pdf",
        "NIST Cybersecurity Framework (CSF) 2.0",
        ["https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"],
        "%PDF",
    ),
    Doc(
        "nist_sp_800-53r5.pdf",
        "NIST SP 800-53 Rev. 5 - Security and Privacy Controls",
        ["https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"],
        "%PDF",
    ),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolicyLens-corpus-fetcher/0.1",
    "Accept": "text/html,application/pdf,*/*;q=0.8",
    "Accept-Language": "en",
}


def _valid(content: bytes, must_contain: str) -> bool:
    if must_contain == "%PDF":
        return content[:5].startswith(b"%PDF")
    return must_contain.lower().encode() in content.lower()


def fetch(doc: Doc, client: httpx.Client) -> tuple[bytes | None, str]:
    for url in doc.urls:
        try:
            headers = {"Accept": doc.accept, "Accept-Language": "eng"} if doc.accept else None
            r = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            print(f"    {url}\n      -> {type(exc).__name__}: {exc}")
            continue
        if r.status_code != 200:
            print(f"    {url}\n      -> HTTP {r.status_code}")
            continue
        if not _valid(r.content, doc.must_contain):
            print(f"    {url}\n      -> unexpected content ({len(r.content)} bytes, maybe a bot-check page)")
            continue
        return r.content, url
    return None, ""


def main() -> int:
    raw = get_settings().raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    failed: list[Doc] = []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        for doc in CORPUS:
            target = raw / doc.filename
            if target.exists() and target.stat().st_size > 10_000:
                print(f"[skip] {doc.filename} already present ({target.stat().st_size // 1024} KB)")
                continue
            print(f"[get ] {doc.title}")
            content, url = fetch(doc, client)
            if content is not None:
                target.write_bytes(content)
                print(f"[ ok ] {doc.filename} ({len(content) // 1024} KB) from {url}")
            else:
                failed.append(doc)

    if failed:
        print("\nSome downloads failed. Please download these by hand into", raw)
        for doc in failed:
            print(f"\n  Save as: {raw / doc.filename}\n  ({doc.title})")
            for url in doc.urls:
                print(f"    {url}")
        print("\nIngestion continues with whatever is present.")
    else:
        print("\nAll corpus documents present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
