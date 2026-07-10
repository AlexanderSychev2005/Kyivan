import os
import glob
import json
import random

def assemble():
    input_files = glob.glob('prepared_datasets/*.jsonl')
    all_docs = []
    
    print("Reading files...")
    for f_path in input_files:
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_docs.append(json.loads(line))
                    
    print(f"Total documents loaded: {len(all_docs)}")
    
    print("Shuffling datasets...")
    random.seed(42)  # For reproducibility
    random.shuffle(all_docs)
    
    out_path = 'data/final_dataset.jsonl'
    print(f"Writing to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for doc in all_docs:
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
    print("Done!")

if __name__ == '__main__':
    assemble()
