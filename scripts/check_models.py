"""Verify that the model IDs in .env exist, by calling each provider's /models endpoint.

    uv run python -m scripts.check_models
"""

import sys

import httpx

from core.config import get_settings


def list_models(base_url: str, key: str) -> list[str]:
    r = httpx.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def main() -> int:
    s = get_settings()
    checks = [
        ("groq", s.groq_api_key, s.groq_base_url, [s.groq_large_model, s.groq_small_model]),
        ("nvidia", s.nvidia_api_key, s.nvidia_base_url, [s.nvidia_large_model, s.nvidia_small_model]),
    ]
    ok = True
    for name, key, url, wanted in checks:
        if not key:
            print(f"[{name}] no API key in .env - skipped")
            ok = ok and name != "groq"
            continue
        try:
            available = list_models(url, key)
        except httpx.HTTPError as exc:
            print(f"[{name}] could not list models: {exc}")
            ok = False
            continue
        print(f"[{name}] {len(available)} models available")
        for model in wanted:
            found = model in available
            ok = ok and found
            print(f"   {'OK ' if found else 'MISSING'} {model}")
        if not all(m in available for m in wanted):
            print("   Some IDs are missing. Candidates:")
            for m in available:
                if any(t in m.lower() for t in ("llama", "gpt-oss", "qwen", "mixtral", "gemma")):
                    print(f"     {m}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
