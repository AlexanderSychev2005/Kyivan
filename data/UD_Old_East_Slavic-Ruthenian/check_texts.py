import json
import csv

with open("ruthenian_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for doc in data:
    if "non_polotsk" in doc["doc_id"]:
        print(f"{doc['doc_id']}: {doc['text'][:100]}")
