import glob
import collections
import re

files = glob.glob('data/pushkin_texts/**/*.txt', recursive=True)

chars = collections.Counter()
brackets = []
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        text = file.read()
        chars.update(text)
        
        # Check for brackets/braces/tags
        if re.search(r'[\{\}\[\]\<\>]', text):
            brackets.append(f)

print('--- Unusual Characters ---')
ALLOWED = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ '
              'ѣꙋѧѳѡꙗєѕІѹꙑѯѥїѠѱӏѫѿꙅꙊꙖꙥѦѭѵѪѻѩӓѸЄ'
              '.,;:-!?+·«»"\'\n\t')
for c, count in chars.most_common():
    if c not in ALLOWED and not c.isdigit() and not (c >= 'a' and c <= 'z') and not (c >= 'A' and c <= 'Z'):
        print(f'{repr(c)} (U+{ord(c):04X}): {count}')

print(f'\nFiles with brackets/braces/tags: {len(brackets)}')
if brackets:
    print('Examples:', brackets[:5])
