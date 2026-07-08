import html
import json
import re


def clean_text(text):
    # Step 0: Unescape HTML entities like &lt; and &gt;
    text = html.unescape(text)

    # Step 1: Remove text reconstruction brackets ( ) [ ] < >
    # This also elegantly solves mixed lacunae like '[... д]' -> '... д'
    text = re.sub(r"[\(\)\[\]\<\>]", "", text)

    # Step 2: Replace lacunae with [UNK]
    # We replace any combination of dots and ellipses that is 2+ chars long, or 1+ ellipses
    # Since we do it exactly at the position of the dots, spaces around it are preserved natively.
    text = re.sub(r"[\.…]{2,}|…+", "[UNK]", text)

    # Optional: Merge multiple consecutive [UNK] tags into one
    # Uncomment the following line if you want to collapse things like '[UNK] [UNK]' into a single '[UNK]'
    # text = re.sub(r'(\[UNK\]\s*){2,}', '[UNK] ', text)

    # Step 3: Cleanup multiple spaces and trailing/leading spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def process_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for doc in data:
        doc["text"] = clean_text(doc["text"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Test with some edge cases
    test_cases = [
        'ковичи ѡпѧт(ь) к Двине. <...>"апор.<...> лофи.',
        "[... д]ѣтми за […] [….] архиеп(и)ск(о)пꙋ Калик[стꙋ",
        "за...тест",
        "+ М(и)л(о)стию Б(о)жою",
        "текст с   лишними пробелами    ",
        "&lt;...&gt;апор.&lt;...&gt;",
    ]

    print("Testing edge cases:")
    for tc in test_cases:
        print(f"Original : '{tc}'")
        print(f"Cleaned  : '{clean_text(tc)}'")
        print("-" * 50)

    # Process the dataset
    input_file = "ruthenian_raw.json"
    output_file = "ruthenian_cleaned.json"

    print(f"\nProcessing dataset from {input_file} to {output_file}...")
    process_dataset(input_file, output_file)
    print("Done!")
