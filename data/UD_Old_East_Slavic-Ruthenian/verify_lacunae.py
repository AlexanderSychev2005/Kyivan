import json
import re

with open(
    "data/UD_Old_East_Slavic-Ruthenian/ruthenian_raw.json", "r", encoding="utf-8"
) as f:
    raw_data = json.load(f)
with open(
    "data/UD_Old_East_Slavic-Ruthenian/ruthenian_cleaned.json", "r", encoding="utf-8"
) as f:
    cleaned_data = json.load(f)

# Find texts that had lacunae
lacuna_regex = re.compile(r"[\.…]{2,}|…+")

md_content = "# Проверка обработки лакун\n\n"
md_content += "| Документ | Оригинал с лакуной | Очищенный текст |\n"
md_content += "|----------|--------------------|-----------------|\n"

found_count = 0

for raw, clean in zip(raw_data, cleaned_data):
    if lacuna_regex.search(raw["text"]):
        orig = raw["text"].replace("\n", " ")
        cln = clean["text"].replace("\n", " ")
        doc_id = raw["doc_id"]
        md_content += f"| `{doc_id}` | {orig} | {cln} |\n"
        found_count += 1

with open(
    r"C:\Users\Alex\.gemini\antigravity\brain\9090b4b2-56d0-4e84-b6dd-24958fdb5e8d\verification.md",
    "w",
    encoding="utf-8",
) as f:
    f.write(md_content)

print(f"Found {found_count} texts with lacunae. Written to verification.md.")
