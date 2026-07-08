import html
import json
import re


def clean_rnc_text(text):
    # 1. Unescape HTML
    text = html.unescape(text)

    # 2. Remove curly braces (e.g. folio numbers {л._54}, {Lib_:_Jud...})
    text = re.sub(r"\{.*?\}", " ", text)

    # 3. Remove HTML-like tags (e.g. <i>, </i>, <em>)
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove [sic] case-insensitive
    text = re.sub(r"\[(?i:sic)\]", " ", text)

    # 5. Keep content but strip parenthesis () and brackets []
    # This restores abbreviations (e.g. д(е)р(е)вни -> деревни) and text (де[нь] -> день)
    text = re.sub(r"[\(\)\[\]]", "", text)

    # 6. Convert lacunae (multiple dots or ellipses) to [UNK]
    text = re.sub(r"[\.…]{2,}|…+", "[UNK]", text)

    # 7. Merge multiple consecutive [UNK] tags into one
    text = re.sub(r"(\[UNK\]\s*){2,}", "[UNK] ", text)

    # 8. Clean multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_rnc_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for doc in data:
        doc["text"] = clean_rnc_text(doc["text"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_file = "rnc_raw.json"
    output_file = "rnc_cleaned.json"

    # Test cases based on RNC
    test_cases = [
        "{л._54}177-г(о) году генваря в 23 де[нь] привез из бѣлевскои д(е)р(е)вни",
        "текст <i>слово</i> [sic] ... и еще .......",
        "{Lib_:_Jud:_Judith.}И вот текст",
    ]

    print("Testing RNC cleaner:")
    for tc in test_cases:
        print(f"Original: {tc}")
        print(f"Cleaned : {clean_rnc_text(tc)}")
        print("-" * 50)

    print(f"Applying cleaning to {input_file}...")
    process_rnc_dataset(input_file, output_file)
    print("Done!")
