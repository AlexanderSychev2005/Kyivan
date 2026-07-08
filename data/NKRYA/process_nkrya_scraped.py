import os
import glob
import json
import re
import unicodedata
from pathlib import Path

# Paths
NKRYA_DIR = Path("C:/Programming/kyivan/data/NKRYA/NKRYA_TEXTS")
RNC_JSON = Path("C:/Programming/kyivan/data/UD_Old_East_Slavic-RNC/rnc_cleaned.json")
OUTPUT_JSON = Path("C:/Programming/kyivan/data/NKRYA/nkrya_scraped_cleaned.json")

# User's cleaning logic adapted
PUNCT_MAP = {
    "†": "+", "×": "+", "*": "+", "⁘": ":", "⁙": ":", "⁞": ":", "¦": ":",
    "∙": "·", ".": "·", "҂": "·", "\uf13f": "·",
}
TITLO_RANGE = range(0x0483, 0x0488)

import html

HOMOGLYPHS = {
    'A': 'А', 'a': 'а', 'B': 'В', 'C': 'С', 'c': 'с', 'E': 'Е', 'e': 'е',
    'H': 'Н', 'K': 'К', 'k': 'к', 'M': 'М', 'O': 'О', 'o': 'о',
    'P': 'Р', 'p': 'р', 'T': 'Т', 'X': 'Х', 'x': 'х', 'y': 'у'
}

def fix_homoglyphs(text):
    def replacer(match):
        word = match.group(0)
        if re.search(r'[а-яА-ЯёЁѣꙋѧѳѡꙗєѕІѹꙑѯѥїѠѱӏѫѿꙅꙊꙖꙥѦѭѵѪѻѩӓѸЄ]', word):
            for lat, cyr in HOMOGLYPHS.items():
                word = word.replace(lat, cyr)
        return word
    return re.sub(r'\b\w+\b', replacer, text)

def clean_text_nkrya(text):
    text = html.unescape(text)
    
    # 0. Fix homoglyphs
    text = fix_homoglyphs(text)
    
    # 1. Remove curly braces (folios, notes)
    text = re.sub(r'\{.*?\}', ' ', text)
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # 3. Remove [sic]
    text = re.sub(r'\[(?i:sic)\]', ' ', text)
    
    # 4. Mask [GAP] and [UNK] so they survive bracket stripping
    text = text.replace('[GAP]', '___GAP___')
    text = text.replace('[UNK]', '___UNK___')
    # Strip parenthesis and brackets
    text = re.sub(r'[\(\)\[\]]', '', text)
    text = text.replace('___GAP___', '[UNK]')
    text = text.replace('___UNK___', '[UNK]')
    
    # 4.5. Process dashes (epigraphy lacunae)
    text = text.replace('‐', '-')
    text = re.sub(r'-{2,}', '[UNK]', text)
    CYR = r'[а-яА-ЯёЁѣꙋѧѳѡꙗєѕІѹꙑѯѥїѠѱӏѫѿꙅꙊꙖꙥѦѭѵѪѻѩӓѸЄ]'
    text = re.sub(f'({CYR})-({CYR})', r'\1[UNK]\2', text)
    text = re.sub(f'({CYR})-({CYR})', r'\1[UNK]\2', text)
    text = re.sub(f'({CYR})-(?!\d)', r'\1[UNK]', text)
    text = re.sub(rf'(?<!\d)-({CYR})', r'[UNK]\1', text)
    
    # 5. Convert lacunae to UNK
    text = re.sub(r'[\.…]{2,}|…+', '[UNK]', text)
    
    # 6. Map historical punctuation
    for old, new in PUNCT_MAP.items():
        text = text.replace(old, new)
        
    # 7. Remove titlos and combining inverted breve (U+0311)
    text = "".join(c for c in text if ord(c) not in TITLO_RANGE and c != '\u0311')
    
    # 8. Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # 9. Merge multiple consecutive UNKs
    text = re.sub(r'(\[UNK\]\s*){2,}', '[UNK] ', text)
    
    # 10. Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_for_dedup(text):
    # Remove all spaces and non-word characters for fuzzy substring matching
    return re.sub(r'[\W_]+', '', text).lower()

def extract_metadata_from_path(path_str):
    folder_name = os.path.basename(os.path.dirname(path_str))
    file_name = os.path.basename(path_str)
    category_match = re.search(r'(DAILY|LEGAL|LIT|SCIENCE)', path_str)
    category = category_match.group(1) if category_match else "unknown"
    
    # Default
    dialect = "unknown"
    year_range = "unknown"
    
    # Extract dialect from folder or file name
    search_str = f"{folder_name} {file_name}"
    dialect_match = re.search(r'(starorus|oldrus|drevnirusskiy|birch|epigraph)', search_str, re.IGNORECASE)
    if dialect_match:
        dialect = dialect_match.group(1).lower()
        
    # Extract year
    year_match = re.search(r'(1\d{3})_(1\d{3})', search_str)
    if year_match:
        year_range = f"{year_match.group(1)}-{year_match.group(2)}"
        
    return dialect, year_range, category

def process_and_deduplicate():
    # 1. Load full RNC texts
    print(f"Loading reference corpus: {RNC_JSON}")
    with open(RNC_JSON, 'r', encoding='utf-8') as f:
        rnc_data = json.load(f)
        
    # Normalize RNC texts for fast substring matching
    rnc_normalized_texts = [normalize_for_dedup(doc['text']) for doc in rnc_data]
    print(f"Loaded {len(rnc_normalized_texts)} reference documents.")

    # 2. Iterate through scraped snippets
    files = glob.glob(str(NKRYA_DIR / "**" / "*.txt"), recursive=True)
    print(f"Found {len(files)} scraped files.")
    
    final_documents = []
    dropped_count = 0
    empty_count = 0
    
    # Deduplication Set (to prevent adding the same snippet multiple times)
    seen_snippets = set()
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        cleaned_text = clean_text_nkrya(raw_text)
        if not cleaned_text or len(cleaned_text) < 15:
            empty_count += 1
            continue
            
        norm_text = normalize_for_dedup(cleaned_text)
        
        # Deduplicate against itself
        if norm_text in seen_snippets:
            dropped_count += 1
            continue
            
        # Deduplicate against RNC full texts
        is_duplicate = False
        for rnc_norm in rnc_normalized_texts:
            if norm_text in rnc_norm:
                is_duplicate = True
                break
                
        if is_duplicate:
            dropped_count += 1
            continue
            
        # Keep snippet
        seen_snippets.add(norm_text)
        dialect, year, category = extract_metadata_from_path(file_path)
        
        doc_id = f"nkrya_scraped_{len(final_documents)}"
        folder_name = os.path.basename(os.path.dirname(file_path))
        final_documents.append({
            "doc_id": doc_id,
            "text": cleaned_text,
            "dialect": dialect,
            "year": year,
            "category": category,
            "source": folder_name
        })
        
    print(f"Processing complete:")
    print(f"  Total scanned: {len(files)}")
    print(f"  Empty/Too short: {empty_count}")
    print(f"  Duplicates dropped: {dropped_count}")
    print(f"  Final unique snippets: {len(final_documents)}")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_documents, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    process_and_deduplicate()
