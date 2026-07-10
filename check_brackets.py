import json
import re

with open('data/sofia/histdict_normalized.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for doc in data[:5]:
    text = doc.get('content', '')
    m = re.findall(r'(.{0,10}[\(\[].{1,10}[\)\]].{0,10})', text)
    if m:
        print(f"Doc {doc.get('doc_id')}:")
        for match in m[:5]:
            print(f'  {match}')
