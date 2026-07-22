"""
Byliny (Russian epic folk poetry) source preparation.

Converts the two `data/byliny/clean_original_*` raw-text folders (already
hand-cleaned of blank lines and footnote markers) into the per-document JSON
list `prepare_datasets.py` expects for a new source.

No date/dialect metadata exists for these texts. Deliberately left unset
here (no "dialect"/"year" keys) rather than guessed -- prepare_datasets.py's
get_macro_dialect() falls back to "Unknown" for an unrecognized dataset_name
and get_date_target(None) now returns a real None, both of which
prepare_splits.py/the collator correctly treat as "withhold this label"
instead of mislabeling every document as OES/bin-0.

`data/byliny/old/` and `data/byliny/process_clean_original.py` are an
earlier, superseded attempt (different raw_texts, produces one combined
blob + a token-count CSV, not per-document JSON) -- not used here.
"""

import json
import os
import re

RAW_DIRS = {
    "novoya": "data/byliny/clean_original_novoya_zapis",
    "staraya": "data/byliny/clean_original_staraya_zapis",
}
OUT_PATH = "data/byliny/byliny.json"


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    return re.sub(r"[ \t]+", " ", text)


def main() -> None:
    dataset = []
    for tag, folder in RAW_DIRS.items():
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()

            text = clean_text(text)
            if not text:
                continue

            dataset.append(
                {
                    "doc_id": f"byliny_{tag}_{os.path.splitext(filename)[0]}",
                    "text": text,
                    "category": "epic_poetry",
                }
            )

    print(f"Docs written: {len(dataset)}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
