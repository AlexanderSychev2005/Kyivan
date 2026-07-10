import json
import csv
import re

with open('ruthenian_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('RUS Project - DIACU.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

def normalize_name(s):
    # Remove numbers at start "002_", remove extensions, lower case
    s = re.sub(r'^\d+_', '', s)
    s = s.replace('.txt', '').lower()
    # Normalize variants
    s = s.replace('polotsk_letters_', 'polotsk__')
    s = s.replace('ratushna_kniga_1986_', 'ratushnakniga_1986__ratush')
    # Remove underscores for fuzzy match
    s = s.replace('_', '').replace('-', '')
    return s

csv_map = {}
for row in csv_rows:
    norm = normalize_name(row['Filename'])
    csv_map[norm] = row

missing = []
matched_count = 0

for doc in data:
    doc_id = doc['doc_id']
    norm_doc = doc_id.lower().replace('_', '').replace('-', '')
    
    # RatushnaKniga has year appended in doc_id sometimes e.g. ratush001653
    # Try stripping trailing numbers for a better match if exact fails
    norm_doc_stripped = re.sub(r'\d{4}$', '', norm_doc)
    
    match = None
    if norm_doc in csv_map:
        match = csv_map[norm_doc]
    elif norm_doc_stripped in csv_map:
        match = csv_map[norm_doc_stripped]
    else:
        # Try substring match as fallback
        for k, v in csv_map.items():
            if k in norm_doc or norm_doc in k:
                match = v
                break
                
    if match:
        doc['dialect'] = match.get('Raw_Language', '').strip()
        doc['year'] = match.get('Target_Year', '').strip()
        
        if not doc['dialect']: doc['dialect'] = 'Ruthenian'
        if not doc['year']: doc['year'] = 'unknown'
        
        matched_count += 1
    else:
        doc['dialect'] = 'Ruthenian'
        doc['year'] = 'unknown'
        missing.append(doc_id)

with open('ruthenian_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Total documents: {len(data)}")
print(f"Matched: {matched_count}")
print(f"Missing: {len(missing)}")

with open('missing_metadata.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(missing))
