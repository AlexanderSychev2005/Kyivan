import json
import re

with open('ruthenian_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

extracted = 0

for doc in data:
    if doc.get('year') == 'unknown':
        # Look for 4 digits starting with 13, 14, 15, or 16 in the doc_id
        match = re.search(r'(1[3456]\d{2})', doc['doc_id'])
        if match:
            doc['year'] = match.group(1)
            extracted += 1
            print(f"Extracted {doc['year']} from {doc['doc_id']}")

with open('ruthenian_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nExtracted year for {extracted} documents.")
