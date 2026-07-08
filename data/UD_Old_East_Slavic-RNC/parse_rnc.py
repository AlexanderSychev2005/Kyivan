import json
import os
from pathlib import Path

INPUT_FILES = [
    "orv_rnc-ud-train.conllu",
    "orv_rnc-ud-dev.conllu",
    "orv_rnc-ud-test.conllu"
]
OUTPUT_JSON = "rnc_raw.json"

def parse_conllu_to_json(file_paths, output_path):
    documents = []
    current_doc = None

    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"File {file_path} not found, skipping...")
            continue
            
        print(f"Processing: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("# newdoc"):
                    if current_doc is not None:
                        current_doc["text"] = " ".join(current_doc["text_sentences"])
                        del current_doc["text_sentences"]
                        documents.append(current_doc)
                        
                    # Extract doc_id
                    if "=" in line:
                        doc_id = line.split("=", 1)[1].strip()
                    else:
                        doc_id = f"unknown_{len(documents)}"
                        
                    current_doc = {
                        "doc_id": doc_id,
                        "lang": "unknown",
                        "split": Path(file_path).stem.split("-")[-1],
                        "text_sentences": []
                    }
                    
                elif line.startswith("# lang =") and current_doc is not None:
                    current_doc["lang"] = line.split("=", 1)[1].strip()
                    
                elif line.startswith("# text =") and current_doc is not None:
                    sentence = line.split("=", 1)[1].strip()
                    current_doc["text_sentences"].append(sentence)
                    
        if current_doc is not None:
            current_doc["text"] = " ".join(current_doc["text_sentences"])
            del current_doc["text_sentences"]
            documents.append(current_doc)
            current_doc = None

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(documents, out_f, ensure_ascii=False, indent=2)
        
    print(f"\nDone! Parsed {len(documents)} documents.")

if __name__ == "__main__":
    parse_conllu_to_json(INPUT_FILES, OUTPUT_JSON)
