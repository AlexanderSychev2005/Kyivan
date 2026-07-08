#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер текстового корпуса и хронографа с https://histdict.uni-sofia.bg

Собирает:
  - 147 текстов из https://histdict.uni-sofia.bg/textcorpus/list
       (страницы вида /textcorpus/show/doc_XXX)
  -   7 текстов из https://histdict.uni-sofia.bg/chronograph/clist
       (страницы вида /chronograph/cshow/doc_XXX)

Сохраняет метаинформацию + сырой текст (БЕЗ нормализации / предобработки)
в один JSON-файл.

Зависимости:
    pip install requests beautifulsoup4 lxml

Запуск:
    python scrape_histdict.py
"""

import json
import re
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://histdict.uni-sofia.bg"

LIST_PAGES = [
    # (url списка, шаблон URL показа документа)
    (f"{BASE}/textcorpus/list", f"{BASE}/textcorpus/show/{{}}"),
    (f"{BASE}/chronograph/clist", f"{BASE}/chronograph/cshow/{{}}"),
]

OUTPUT_PATH = Path("histdict_corpus.json")

REQUEST_DELAY = 1.0        # пауза между запросами, сек (вежливость к серверу)
TIMEOUT = 60
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0"
    ),
    "Accept-Language": "bg,ru;q=0.8,en;q=0.5",
}

# --- Метаполя, которые встречаются в блоке метаданных страницы документа.
# Ключ в HTML (болгарская подпись) -> ключ в JSON.
# Список исчерпывающий для обоих разделов; отсутствующие поля просто
# не попадут в результат для конкретного документа.
META_LABELS = {
    "Заглавие на латински": "latin_title",
    "Жанр": "genre",
    "Автор": "author",
    "Дата на ръкописа": "manuscript_date",
    "Дата на превода": "translation_date",
    "Дата на преписа": "copy_date",
    "Правопис": "orthography",
    "Име на ръкописа": "manuscript_name",
    "Хранилище на ръкописа": "manuscript_repository",
    "Сигнатура на ръкописа": "manuscript_signature",
    "Страници": "pages",
    "doc_id": "doc_id",
}

# Порядок проб — длинные подписи первыми, чтобы "Дата на ръкописа" не
# перехватывалась внутри других строк.
META_LABELS_ORDERED = sorted(META_LABELS.keys(), key=len, reverse=True)


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get(session, url):
    """GET с ретраями. Гарантирует корректную кодировку UTF-8."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            # сайт отдаёт UTF-8; форсируем, чтобы не было mojibake
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            last_exc = e
            wait = REQUEST_DELAY * attempt * 2
            print(f"  ! попытка {attempt}/{MAX_RETRIES} не удалась для {url}: {e}"
                  f" — жду {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise last_exc


def extract_doc_ids(html):
    """
    Из HTML страницы-списка вытаскивает id документов (doc_XXX)
    по ссылкам на show/cshow. Сохраняет порядок и убирает дубли.
    """
    soup = BeautifulSoup(html, "lxml")
    ids = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/(?:show|cshow)/(doc_\d+)", a["href"])
        if m:
            doc_id = m.group(1)
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
    return ids


def _clean(s):
    """Схлопывает пробелы/переводы строк ТОЛЬКО в метазначениях (не в тексте)."""
    return re.sub(r"\s+", " ", s).strip(" :\u00a0\t\n·")


def _is_inside(el, class_name):
    p = el
    while p is not None:
        cls = p.get("class") or []
        if class_name in cls:
            return True
        p = getattr(p, "parent", None)
    return False

# --- Тело памятника: извлечь только основной текст из real_text/body ---
def _extract_main_from_spans(soup):
    root = soup.select_one("div.real_text div.body")
    if root is None:
        return ""

    parts = []
    # обработаем по <p>, чтобы сохранять логические параграфы
    for p in root.find_all("p"):
        # для каждого прямого ребёнка параграфа (сохраняем порядок)
        for node in p.contents:
            # NavigableString (текстовый узел)
            if getattr(node, "name", None) is None:
                # добавляем текст как есть (включая пробелы внутри)
                parts.append(str(node))
                continue

            # Тэги
            name = node.name.lower()
            # <br> — перевод строки внутри параграфа
            if name == "br":
                parts.append("\n")
                continue

            # интересуют span'ы (и другие теги — берем их текст)
            if name == "span":
                cls = set(node.get("class", []))
                # пропускаем редакторские метки и номера страниц
                if "note_comment" in cls or "pagenum" in cls:
                    continue
                # включаем всё остальное (в т.ч. note_diff)
                # не strip'им — сохраняем ведущие/замыкающие пробелы, они значимы
                parts.append(node.get_text(strip=False))
            else:
                # для прочих тэгов (если встретятся) — берем их текст
                parts.append(node.get_text(strip=False))

        # конец параграфа — добавляем перевод строки
        parts.append("\n")

    # соберём в строку, затем почистим лишние пробелы внутри каждой строки
    text = "".join(parts)

    # заменим «много пустых строк» на одну, и схлопнём повторяющие пробелы в строках
    lines = [re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in text.splitlines()]
    # удаляем пустые строки в начале/конце, сохраняем межстрочные переносы
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def parse_document(html, url, doc_id):
    """
    Разбирает страницу документа.

    Возвращает dict с метаданными и полем 'content' (сырой текст тела,
    БЕЗ нормализации — сохраняются оригинальные символы, точки-разделители,
    надстрочные знаки, маркеры страниц и т.п.).
    """
    soup = BeautifulSoup(html, "lxml")

    # Заголовок (кириллическое заглавие) — обычно первый <h*> после
    # навигации, либо <title> вида "Text Corpus - <latin>".
    title = None
    for tag in soup.find_all(["h1", "h2", "h3"]):
        t = tag.get_text(strip=True)
        if t and t not in ("Текстов корпус", "Архивски Хронограф"):
            title = t
            break

    # Забираем весь видимый текст основного контейнера.
    # На страницах histdict основное содержимое лежит в блоке контента;
    # надёжнее всего взять body и затем отрезать навигацию/подвал.
    body = soup.find("body") or soup
    full_text = body.get_text("\n")

    # --- Отрезаем навигационное меню (до "Текстов корпус"/"Архивски Хронограф")
    for marker in ("Текстов корпус", "Архивски Хронограф"):
        idx = full_text.find(marker)
        if idx != -1:
            full_text = full_text[idx + len(marker):]
            break

    # --- Отрезаем подвал с копирайтом
    copyr = full_text.find("© Софийски университет")
    if copyr != -1:
        full_text = full_text[:copyr]

    # Нормализуем только переводы строк для удобства разбора метаблока,
    # но исходную версию тела сохраняем отдельно.
    lines = [ln for ln in (l.rstrip() for l in full_text.split("\n"))]

    # --- Разбор метаданных ---
    # Метаблок состоит из подписей "Label: value", которые в рендере идут
    # подряд. В HTML это, как правило, отдельные строки или пары.
    meta = {}
    # Склеим строки метаблока в одну для устойчивого regex-поиска пар,
    # но остановимся на первой "длинной" строке текста (тело памятника).
    joined = "\n".join(lines)

    # Найдём границу: метаданные заканчиваются на 'doc_id...doc_XXX'
    # (это последнее метаполе на странице textcorpus). Для хронографа
    # doc_id может отсутствовать в теле — тогда режем по первому маркеру
    # страницы (например "9b") или просто берём первые ~15 строк как мету.
    meta_zone = joined
    m_docid = re.search(r"doc_id\s*doc_\d+", joined)
    if m_docid:
        meta_zone = joined[: m_docid.end()]
        content_zone = joined[m_docid.end():]
    else:
        # запасной вариант: мета — до первого маркера страницы вида "12a"/"3b"
        m_page = re.search(r"\n\s*\d+[ab]\s*\n", joined)
        if m_page:
            meta_zone = joined[: m_page.start()]
            content_zone = joined[m_page.start():]
        else:
            # совсем запасной: первые 1200 символов — мета, остальное тело
            meta_zone = joined[:1200]
            content_zone = joined[1200:]

    # Извлекаем пары "подпись: значение" из meta_zone.
    # Значение — всё до следующей известной подписи.
    labels_pattern = "|".join(re.escape(l) for l in META_LABELS_ORDERED)
    # регэксп: (подпись)(разделитель)(значение до следующей подписи или конца)
    for m in re.finditer(
        rf"({labels_pattern})\s*[:\uFF1A]?\s*(.*?)(?=(?:{labels_pattern})\s*[:\uFF1A]?|\Z)",
        meta_zone,
        flags=re.DOTALL,
    ):
        label = m.group(1)
        value = _clean(m.group(2))
        key = META_LABELS[label]
        # doc_id: оставляем только сам идентификатор
        if key == "doc_id":
            dm = re.search(r"doc_\d+", value)
            value = dm.group(0) if dm else value
        if value:
            meta[key] = value

    # --- Тело памятника: только основной текст из span'ов real_text/body ---
    content = _extract_main_from_spans(soup)
    if not content:
        content = content_zone.strip("\n")

    # fallback на старую логику, если верстка неожиданная
    if not content:
        content = content_zone.strip("\n")

    record = {
        "doc_id": doc_id,
        "url": url,
        "title": title,
        **meta,
        "content": content,
    }
    return record


def scrape():
    session = make_session()
    all_records = []
    seen_ids = set()

    for list_url, show_tpl in LIST_PAGES:
        print(f"\n=== Список: {list_url} ===")
        list_html = get(session, list_url)
        doc_ids = extract_doc_ids(list_html)
        print(f"Найдено ссылок на документы: {len(doc_ids)}")

        for i, doc_id in enumerate(doc_ids, 1):
            # doc_id из textcorpus и chronograph могут пересекаться (например
            # doc_28 есть и там, и там), поэтому ключом уникальности делаем
            # (section, doc_id) через сам URL.
            doc_url = show_tpl.format(doc_id)
            uniq_key = doc_url
            if uniq_key in seen_ids:
                continue
            seen_ids.add(uniq_key)

            print(f"[{i}/{len(doc_ids)}] {doc_id} -> {doc_url}")
            try:
                html = get(session, doc_url)
                record = parse_document(html, doc_url, doc_id)
                # добавим, из какого раздела пришёл документ
                record["section"] = (
                    "chronograph" if "chronograph" in doc_url else "textcorpus"
                )
                all_records.append(record)
            except Exception as e:
                print(f"  !! пропущен {doc_id}: {e}", file=sys.stderr)
                all_records.append({
                    "doc_id": doc_id,
                    "url": doc_url,
                    "section": "chronograph" if "chronograph" in doc_url else "textcorpus",
                    "error": str(e),
                })

            time.sleep(REQUEST_DELAY)

    return all_records


def main():
    records = scrape()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    ok = [r for r in records if "error" not in r]
    err = [r for r in records if "error" in r]
    print(f"\nГотово. Всего записей: {len(records)} "
          f"(успешно: {len(ok)}, с ошибкой: {len(err)})")
    print(f"Сохранено в: {OUTPUT_PATH.resolve()}")
    if err:
        print("Документы с ошибками:", ", ".join(r["doc_id"] for r in err))


if __name__ == "__main__":
    main()