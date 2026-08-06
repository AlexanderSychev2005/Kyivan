"""
Scraper for "Библиотека литературы Древней Руси" (БЛДР), mirrored on
azbyka.ru. Each work page shows a modern-Russian intro/translation followed
by the original Old Russian / Church Slavonic text, separated by
`<h4 class="title h4">* * *</h4>` marker(s) and closed off by a
"Примечания" (footnotes) section. We want only the *original* text: the
segment after the LAST star marker, up to "Примечания".

Output: one raw JSON per work in raw/tomN_M.json, plus a combined
bldr_all.jsonl.
"""

import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://azbyka.ru/otechnik/Istorija_Tserkvi/biblioteka-literatury-drevnej-rusi-tom-{n}/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = Path(__file__).parent / "raw"
STAR = '<h4 class="title h4">* * *</h4>'
# Pre-reform / Old Russian orthography markers -- used as a sanity check
# that the extracted segment really is the original, not another intro.
OLD_ORTHO_RE = re.compile(r"[ѣѳѵіѫѧѩѭꙗѡѹ]", re.IGNORECASE)


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def strip_tags(segment: str) -> str:
    # Drop footnote note-blocks and the "notes" heading itself if they
    # leaked into the segment, then strip all remaining tags.
    segment = re.sub(r'<div class="note">.*?</div>', " ", segment, flags=re.S)
    segment = re.sub(r"<sup>\d+</sup>", "", segment)
    segment = re.sub(r"<[^>]+>", " ", segment)
    segment = html.unescape(segment)
    segment = re.sub(r"[ \t]+", " ", segment)
    segment = re.sub(r"\n\s*\n+", "\n", segment)
    return segment.strip()


H2 = '<h2 class="text-center">'
# Volumes >=10 stop using the "* * *" star divider for some works; a few
# instead pair a modern-title h2 with an old-orthography-title h2 right
# before the original (h2 count == 2), and 16th-17th century works are
# often presented as a single text with NO parallel translation at all
# (orthography by then is close enough to modern that OLD_ORTHO_RE won't
# fire) -- gated on the TOC title carrying a primary-source genre word so
# pure scholarly essays (e.g. "«Смеховой мир» Древней Руси") don't get
# swept in as if they were period text.
PRIMARY_GENRE_WORDS = (
    "Повесть", "Житие", "Сказание", "Послание", "Слово", "Хождение",
    "Грамота", "Переписка", "Записка", "Сочинени", "Плач", "Челобитная",
    "Поучение", "Моление", "Похвала", "Явление", "Чудеса", "Служба",
    "Духовная", "Устав", "Правда", "Летопис", "Беседа", "Послания",
)


TAIL_MARKERS = ("Telegram-каналы", "Вам может быть интересно", "Источник:")


def extract_original(page_html: str, toc_title: str = "") -> tuple[str | None, str]:
    book_idx = page_html.find('class="book"')
    if book_idx == -1:
        return None, "no-book-div"
    # page_html.find lands on the *attribute*, not the tag start -- slicing
    # from here left "class="book">" itself as leading literal text in
    # every single-text (no star/h2 marker) page, since that branch takes
    # book_html[:content_end] rather than starting after a found marker.
    tag_close = page_html.find(">", book_idx)
    book_html = page_html[tag_close + 1 :] if tag_close != -1 else page_html[book_idx:]

    stop_candidates = [
        pos
        for pos in (
            book_html.find("Примечания"),
            *(book_html.find(m) for m in TAIL_MARKERS),
        )
        if pos != -1
    ]
    # A page with no footnotes section falls through to
    # `end = len(book_html)`, which then also swallows azbyka's own
    # related-content widget ("Вам может быть интересно...") and its
    # "Telegram-каналы" link -- neither is part of the work's text.
    content_end = min(stop_candidates) if stop_candidates else len(book_html)

    stars = [m.start() for m in re.finditer(re.escape(STAR), book_html) if m.start() < content_end]
    if stars:
        start = stars[-1] + len(STAR)
        segment = strip_tags(book_html[start:content_end])
        if len(segment) < 200 or not OLD_ORTHO_RE.search(segment):
            return None, "star-failed-orthocheck"
        return segment, "star"

    h2s = [m.start() for m in re.finditer(re.escape(H2), book_html) if m.start() < content_end]
    if len(h2s) == 2:
        segment = strip_tags(book_html[h2s[-1]:content_end])
        if len(segment) < 200:
            return None, "h2pair-too-short"
        return segment, "h2pair"
    if len(h2s) >= 4 and len(h2s) % 2 == 0:
        # Some single-work, chapter-numbered compilations (verified by hand
        # on "Домострой", tom10/6, 130 h2s) lay out ALL chapters in modern
        # translation first, then repeat the SAME chapter count in original
        # orthography starting the numbering over at "1." -- confirmed by
        # the second half's chapter titles echoing the first half's
        # ("Наставление отца сыну" -> "Наказание от отца к сыну", etc).
        # Multi-work anthologies (e.g. Avvakum's collected works, 44 h2s of
        # distinct titles) don't restart numbering and fall through to
        # needs-review below rather than getting force-split in half.
        half = len(h2s) // 2

        def chapter_1_text(pos: int) -> str:
            m = re.search(r"<h2 class=\"text-center\">\s*(.{0,60})", book_html[pos:])
            return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))) if m else ""

        first_ch1 = chapter_1_text(h2s[1])
        second_ch1 = chapter_1_text(h2s[half + 1]) if half + 1 < len(h2s) else ""
        restarts = first_ch1.strip().startswith("1.") and second_ch1.strip().startswith("1.")
        if restarts:
            segment = strip_tags(book_html[h2s[half]:content_end])
            if len(segment) >= 500:
                return segment, f"h2-half-split-{len(h2s)}"
        return None, f"compilation-{len(h2s)}-h2-needs-review"

    if not any(w in toc_title for w in PRIMARY_GENRE_WORDS):
        return None, "no-genre-word-likely-essay"
    segment = strip_tags(book_html[:content_end])
    if len(segment) < 800:
        return None, "too-short"
    return segment, "single-text"


def get_title(page_html: str) -> str:
    m = re.search(r"<h1>\s*(.*?)\s*(?:<br>)?\s*</h1>", page_html, re.S)
    if not m:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()


def get_toc(volume: int) -> list[tuple[int, str]]:
    toc_html = fetch(BASE.format(n=volume))
    if not toc_html:
        return []
    idx = toc_html.find("Содержание")
    if idx == -1:
        return []
    end = toc_html.find('<a id="0_1">', idx)
    section = toc_html[idx: end if end != -1 else idx + 20000]
    pairs = re.findall(r'href="\./(\d+)"><span class="h2o">\s*([^<]+)</span>', section)
    return [(int(n), html.unescape(t).strip()) for n, t in pairs]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for volume in range(1, 21):
        toc = get_toc(volume)
        print(f"tom {volume}: {len(toc)} works")
        for work_num, toc_title in toc:
            out_path = OUT_DIR / f"tom{volume}_{work_num}.json"
            if out_path.exists():
                continue
            url = f"{BASE.format(n=volume)}{work_num}"
            page_html = fetch(url)
            if not page_html:
                print(f"  [{volume}/{work_num}] fetch failed, skip")
                continue
            text, reason = extract_original(page_html, toc_title)
            title = get_title(page_html) or toc_title
            if not text:
                print(f"  [{volume}/{work_num}] {title!r}: skip ({reason})")
                time.sleep(0.4)
                continue
            record = {
                "volume": volume,
                "work_num": work_num,
                "title": title,
                "url": url,
                "text": text,
            }
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            combined.append(record)
            print(f"  [{volume}/{work_num}] {title!r}: {len(text)} chars")
            time.sleep(0.4)

    # Combined jsonl also picks up any records from a previous partial run
    # that are already sitting in raw/ (out_path.exists() skip above).
    all_records = []
    for p in sorted(OUT_DIR.glob("tom*.json")):
        all_records.append(json.loads(p.read_text(encoding="utf-8")))
    combined_path = Path(__file__).parent / "bldr_all.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nTotal works saved: {len(all_records)} -> {combined_path}")


if __name__ == "__main__":
    main()
