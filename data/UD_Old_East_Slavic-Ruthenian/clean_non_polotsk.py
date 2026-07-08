import json
import re
import html

def standard_clean(text):
    # Step 0: Unescape HTML entities
    text = html.unescape(text)
    
    # 0.5 Literal slashes indicating line/page breaks in manuscript (e.g. //, /)
    text = re.sub(r'//|/', '', text)
    
    # Step 1: Remove text reconstruction brackets
    text = re.sub(r'[\(\)\[\]\<\>]', '', text)
    
    # Step 2: Replace lacunae with [UNK]
    text = re.sub(r'[\.…]{2,}|…+', '[UNK]', text)
    
    # Merge multiple consecutive [UNK] tags into one
    text = re.sub(r'(\[UNK\]\s*){2,}', '[UNK] ', text)
    
    # Step 3: Cleanup multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def advanced_clean_non_polotsk(text):
    # Unescape HTML first
    text = html.unescape(text)
    
    # REMOVE EXTRA NOISE BEFORE STRIPPING BRACKETS
    # 1. Curly braces and their contents (e.g. {22_зв.}, {76})
    text = re.sub(r'\{.*?\}', ' ', text)
    
    # 2. Signature marks
    text = re.sub(r'\[?Знак підпису\.?\]?', ' ', text)
    
    # 3. Folio/page marks in Ukrainian (e.g. [1_зв.], 4_зв.)
    text = re.sub(r'\[?\d+_зв\.?\]?', ' ', text)
    
    # 4. Folio/page marks in Russian (e.g. [412 об.])
    text = re.sub(r'\[?\d+\s*об\.?\]?', ' ', text)
    
    # 5. Document numbering (e.g. № 11., [№ 1.])
    text = re.sub(r'\[?№\s*\d+\.?\]?', ' ', text)
    
    # 6. Editorial exclamation marks (e.g. [!])
    text = re.sub(r'\[!\]', ' ', text)
    
    # 7. Numbered lists in brackets (e.g. [1.], [2.])
    text = re.sub(r'\[\d+\.\]', ' ', text)
    
    # 7.5 Arabic numeral copies in brackets (e.g. [1652], [18])
    text = re.sub(r'\[\d+\]', ' ', text)
    
    # 8. Line breaks and weird editorial tags in the original XML (e.g. <_>, <_.>, <⌒>, <⋮>, <lbr/>)
    text = re.sub(r'<(_|_\.|⌒|⋮|lbr/)>', '', text)
    
    # 9. Literal slashes indicating line/page breaks in manuscript (e.g. //, /)
    text = re.sub(r'//|/', '', text)
    
    # Now proceed with the standard cleaning on the resulting text
    text = re.sub(r'[\(\)\[\]\<\>]', '', text)
    text = re.sub(r'[\.…]{2,}|…+', '[UNK]', text)
    
    # Merge multiple consecutive [UNK] tags into one
    text = re.sub(r'(\[UNK\]\s*){2,}', '[UNK] ', text)
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for doc in data:
        if 'RatushnaKniga' in doc['doc_id']:
            # Search for a 4-digit number in brackets starting with 15, 16, or 17
            match = re.search(r'\[(1[567]\d{2})\]', doc['text'])
            if match:
                doc['doc_id'] = f"{doc['doc_id']}_{match.group(1)}"
                
        if doc['doc_id'].startswith('polotsk'):
            doc['text'] = standard_clean(doc['text'])
        else:
            doc['text'] = advanced_clean_non_polotsk(doc['text'])
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    input_file = 'ruthenian_raw.json'
    output_file = 'ruthenian_cleaned.json'
    
    print(f"Applying advanced cleaning to non-Polotsk documents...")
    process_dataset(input_file, output_file)
    print("Done!")
