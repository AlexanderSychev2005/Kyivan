"""
Scraper for litopys.org.ua (Izbornyk) -- Lavrentian and Ipatian chronicles,
diplomatic edition (correct pre-reform Cyrillic, unlike OCR'd PSRL scans).

Unlike the azbyka.ru BLDR scrape, this source interleaves base chronicle
text with critical-apparatus blocks ("Варіанты" -- manuscript variant
readings, "Примечания" -- scribal/paleographic notes) inline, without a
clean tag boundary between them. We only strip HTML/boilerplate here and
leave the apparatus in place with a `needs_cleaning: true` flag -- trying
to regex out the apparatus without visual verification risks silently
corrupting the base text, which is worse than shipping it uncleaned but
correctly labeled.
"""

import html
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {"User-Agent": "Mozilla/5.0"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
OUT_DIR = Path(__file__).parent / "raw"

CHRONICLES = {
    "lavrentian": ("https://litopys.org.ua/lavrlet/{p}.htm", "lavr", 34),
    "ipatian": ("https://litopys.org.ua/ipatlet/{p}.htm", "ipat", 41),
}

NAV_PATTERNS = [
    r"Уклінно просимо.*?Опитування про фемінативи",
    r"гостьова.*?пошук",
    r"ІЗБОРНИК.*?ІЗБОРНИК",
    r"ЛІТОПИСИ.*?ЛЕКСИКОНИ",
    r"Шрифт\s*(Попередня|Наступна|Зміст)?",
    r"Попередня\s*Зміст\s*Наступна",
    r"Якщо пом[іi]тили пом[иі]лку.*?Ctrl\+Enter\.",
    r"\[ПСРЛ\.[^\]]*\]",  # per-page bibliographic citation, e.g. "[ПСРЛ. - Т. 1. ... - Стлб. 29-46.]"
    r"\bЗміст\b",
    r"\bНаступна\b",
    r"\bПопередня\b",
]

# Arabic digits are never genuine chronicle content -- years in the base
# text are always spelled with letter-numerals + titlo (e.g. "҂s҃.у҃.еı҃."),
# so any digit is either a footnote-reference marker (bare "4"), a folio
# marker ("/л.2об./"), or a column marker before "Примѣчанія" ("29п") --
# all apparatus (see module docstring), never part of the text itself. A
# word-boundary-gated version missed the latter two (digit fused directly
# onto the adjacent Cyrillic abbreviation, no separator to anchor on), so
# this strips digits unconditionally rather than chasing each new fused
# variant one at a time.
_FOOTNOTE_NUM_RE = re.compile(r"\d+")


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20, context=CTX) as r:
                return r.read().decode("windows-1251", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def clean(page_html: str) -> str:
    body_idx = page_html.find("<body")
    text = page_html[body_idx:] if body_idx != -1 else page_html
    # SCRIPT/STYLE tags on this site are uppercase -- a lowercase-only
    # pattern silently missed the whole inline error-report <SCRIPT> block
    # (an onkeypress handler + a hidden form), leaking "var is_ok = false;
    # document.onkeypress=..." etc. straight into the scraped text.
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    for pat in NAV_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.S)
    text = _FOOTNOTE_NUM_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for name, (url_tml, prefix, max_n) in CHRONICLES.items():
        for n in range(1, max_n + 1):
            page_id = f"{prefix}{n:02d}"
            out_path = OUT_DIR / f"{name}_{page_id}.json"
            if out_path.exists():
                combined.append(json.loads(out_path.read_text(encoding="utf-8")))
                continue
            url = url_tml.format(p=page_id)
            page_html = fetch(url)
            if not page_html or len(page_html) < 500:
                print(f"[{name}/{page_id}] fetch failed or empty, skip")
                continue
            text = clean(page_html)
            if len(text) < 300:
                print(f"[{name}/{page_id}] too short after cleaning, skip")
                continue
            record = {
                "chronicle": name,
                "page_id": page_id,
                "url": url,
                "text": text,
                "needs_cleaning": True,
            }
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            combined.append(record)
            print(f"[{name}/{page_id}]: {len(text)} chars")
            time.sleep(0.5)

    combined_path = Path(__file__).parent / "litopys_all.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nTotal pages saved: {len(combined)} -> {combined_path}")


if __name__ == "__main__":
    main()
