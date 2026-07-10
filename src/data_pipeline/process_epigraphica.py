import csv
import json
import re

def fingerprint(text):
    # Keep only Cyrillic letters for matching
    return re.sub(r'[^а-яА-ЯёЁѣꙋѧѳѡꙗєѕІѹꙑѯѥїѠѱӏѫѿꙅꙊꙖꙥѦѭѵѪѻѩӓѸЄ]', '', text).lower()

with open('data/epigraphica/epigraphica_full_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

with open('data/epigraphica/epigraphica_final_cleaned.txt', 'r', encoding='utf-8') as f:
    txt_lines = [line.strip() for line in f if line.strip()]

dataset = []
unmatched = 0

for idx, line in enumerate(txt_lines):
    clean_text = line.replace('[CTX_CHURCH]', '').strip()
    # Replace GAP with UNK as per our standard
    final_text = clean_text.replace('[GAP]', '[UNK]')
    
    fp_txt = fingerprint(clean_text)
    
    best_match = None
    best_score = 0
    
    for row in csv_rows:
        fp_csv = fingerprint(row['text'])
        # Find Longest Common Subsequence or simple containment
        if not fp_csv or not fp_txt:
            continue
        
        # simple heuristic: check how many characters from fp_txt are in fp_csv in order
        if fp_txt in fp_csv or fp_csv in fp_txt:
            best_match = row
            break
            
    if not best_match:
        # Fallback to Jaccard similarity of 3-grams
        def ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        txt_grams = ngrams(fp_txt)
        if txt_grams:
            for row in csv_rows:
                fp_csv = fingerprint(row['text'])
                csv_grams = ngrams(fp_csv)
                if not csv_grams: continue
                
                intersect = len(txt_grams & csv_grams)
                union = len(txt_grams | csv_grams)
                score = intersect / union if union > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_match = row
                    
    year = 'unknown'
    doc_id = f'epigraphica_{idx}'
    
    if best_match:
        year = best_match.get('date', 'unknown')
        if not year.strip(): year = 'unknown'
        row_id = best_match.get('\ufeffid', best_match.get('id', str(idx)))
        doc_id = f'epigraphica_{row_id}'
    else:
        unmatched += 1
        
    dataset.append({
        'doc_id': doc_id,
        'text': final_text,
        'dialect': 'epigraph',
        'year': year,
        'category': 'DAILY'
    })

print(f'Total TXT lines: {len(txt_lines)}')
print(f'Unmatched: {unmatched}')

with open('data/epigraphica/epigraphica.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print('Saved to data/epigraphica/epigraphica.json')
