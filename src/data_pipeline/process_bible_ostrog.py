import glob
import json
import os
import re

RAW_DIR = 'data/bible_ostrog/raw'
OUT_PATH = 'data/bible_ostrog/bible_ostrog.json'
YEAR = '1581'


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, '*.txt')))
    dataset = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name == 'char_map':
            continue

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue

        dataset.append({
            'doc_id': f'bible_ostrog_{name}',
            'text': text,
            'dialect': 'church_slavonic',
            'year': YEAR,
            'category': 'RELIGIOUS',
            'source': name,
        })

    print(f'Books written: {len(dataset)}')

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
