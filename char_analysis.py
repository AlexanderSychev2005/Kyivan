import json
import glob
from collections import Counter
import unicodedata

files = glob.glob('prepared_datasets/*.jsonl')
char_counts = Counter()

for f_path in files:
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            text = doc.get('text', '')
            char_counts.update(text)

cyrillic = {}
latin = {}
numbers = {}
punctuation = {}
other = {}

for char, count in char_counts.most_common():
    try:
        name = unicodedata.name(char)
    except:
        name = 'UNKNOWN'
        
    if 'CYRILLIC' in name:
        cyrillic[char] = count
    elif 'LATIN' in name:
        latin[char] = count
    elif char.isdigit():
        numbers[char] = count
    elif unicodedata.category(char).startswith('P'):
        punctuation[char] = count
    else:
        other[char] = count

print('--- CYRILLIC (Rare/Specific) ---')
for c, cnt in list(cyrillic.items())[-40:]: 
    try: name = unicodedata.name(c)
    except: name = 'UNK'
    print(f"{repr(c)} ({name}): {cnt}")

print('\n--- LATIN ---')
for c, cnt in latin.items(): print(f'{repr(c)}: {cnt}', end=', ')

print('\n\n--- PUNCTUATION ---')
for c, cnt in punctuation.items(): print(f'{repr(c)}: {cnt}', end=', ')

print('\n\n--- OTHER ---')
for c, cnt in other.items(): print(f'{repr(c)}: {cnt}', end=', ')
