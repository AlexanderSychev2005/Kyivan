import json
import csv
import glob
import os

# --- Pushkin Texts ---
print("Processing Pushkin texts...")
pushkin_csv = 'data/pushkin_texts/pushkin_texts_data.csv'
pushkin_files = glob.glob('data/pushkin_texts/**/*.txt', recursive=True)

pushkin_meta = {}
with open(pushkin_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row.get('Filename', '').strip()
        if filename:
            pushkin_meta[filename] = row

pushkin_dataset = []
for file_path in pushkin_files:
    filename = os.path.basename(file_path)
    
    # Extract category from parent directory
    category = os.path.basename(os.path.dirname(file_path))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        
    meta = pushkin_meta.get(filename, {})
    
    year = meta.get('Year number', '').strip()
    if not year:
        year = meta.get('Raw Year', 'unknown').strip()
        
    dialect = meta.get('Raw Dialect', '').strip()
    if not dialect:
        dialect = 'Old Russian'
        
    pushkin_dataset.append({
        'doc_id': f"pushkin_{filename.replace('.txt', '')}",
        'text': text,
        'dialect': dialect,
        'year': year,
        'category': category
    })

with open('data/pushkin_texts/pushkin_texts.json', 'w', encoding='utf-8') as f:
    json.dump(pushkin_dataset, f, ensure_ascii=False, indent=2)

print(f"Saved {len(pushkin_dataset)} Pushkin documents.")

# --- TOROT Texts ---
print("\nProcessing TOROT texts...")
torot_csv = 'data/TOROT/torot_data.csv'
torot_files = glob.glob('data/TOROT/**/*.txt', recursive=True)

torot_meta = {}
with open(torot_csv, 'r', encoding='utf-8') as f:
    # Handle BOM in first column
    reader = csv.reader(f)
    header = next(reader)
    # create dict reader mapping
    dict_reader = csv.DictReader(f, fieldnames=header)
    for row in dict_reader:
        path = row.get('Path', '').strip().replace('\\', '/')
        if path:
            filename = os.path.basename(path)
            torot_meta[filename] = row

torot_dataset = []
for file_path in torot_files:
    filename = os.path.basename(file_path)
    
    # Extract category from parent directory
    # If the folder is 'torot_LIT', remove 'torot_'
    parent_dir = os.path.basename(os.path.dirname(file_path))
    category = parent_dir.replace('torot_', '')
    if category == 'TOROT': category = 'UNKNOWN'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        
    meta = torot_meta.get(filename, {})
    
    year = meta.get('Year number', '').strip()
    if not year:
        year = meta.get('Year Raw', 'unknown').strip()
        
    dialect = meta.get('Raw Dialect', '').strip()
    if not dialect:
        dialect = meta.get('Language', 'Old Russian').strip()
        
    torot_dataset.append({
        'doc_id': f"torot_{filename.replace('.txt', '')}",
        'text': text,
        'dialect': dialect,
        'year': year,
        'category': category
    })

with open('data/TOROT/torot.json', 'w', encoding='utf-8') as f:
    json.dump(torot_dataset, f, ensure_ascii=False, indent=2)

print(f"Saved {len(torot_dataset)} TOROT documents.")
