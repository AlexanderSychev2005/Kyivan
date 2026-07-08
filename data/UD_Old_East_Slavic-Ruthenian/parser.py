#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# Список исходных файлов
INPUT_FILES = [
    "orv_ruthenian-ud-train.conllu",
    "orv_ruthenian-ud-dev.conllu",
    "orv_ruthenian-ud-test.conllu",
]

OUTPUT_JSON = "ruthenian_raw.json"


def parse_conllu_to_json(file_paths, output_path):
    documents = []
    current_doc = None

    for file_path in file_paths:
        # Проверяем, существует ли файл, чтобы избежать ошибок
        if not Path(file_path).exists():
            print(f"Файл {file_path} не найден, пропускаем...")
            continue

        print(f"Обработка файла: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Ищем начало нового документа
                if line.startswith("# newdoc"):
                    # Если у нас уже был открыт документ, сохраняем его перед созданием нового
                    if current_doc is not None:
                        # Склеиваем все предложения в один сплошной текст
                        current_doc["text"] = " ".join(current_doc["text_sentences"])
                        del current_doc["text_sentences"]  # Удаляем временный список
                        documents.append(current_doc)

                    # Извлекаем doc_id, обрабатывая разные форматы:
                    # "# newdoc id =", "# newdoc =", "# newdoc_id ="
                    if "=" in line:
                        doc_id = line.split("=", 1)[1].strip()
                    else:
                        doc_id = f"unknown_{len(documents)}"

                    current_doc = {
                        "doc_id": doc_id,
                        "lang": "unknown",  # Значение по умолчанию, если тега lang не будет
                        "split": Path(file_path).stem.split("-")[
                            -1
                        ],  # Вытащит train, dev или test
                        "text_sentences": [],
                    }

                # Ищем тег языка
                elif line.startswith("# lang =") and current_doc is not None:
                    current_doc["lang"] = line.split("=", 1)[1].strip()

                # Ищем оригинальный текст предложения
                elif line.startswith("# text =") and current_doc is not None:
                    sentence_text = line.split("=", 1)[1].strip()
                    current_doc["text_sentences"].append(sentence_text)

        # Не забываем сохранить последний документ из файла
        if current_doc is not None:
            current_doc["text"] = " ".join(current_doc["text_sentences"])
            del current_doc["text_sentences"]
            documents.append(current_doc)
            # Сбрасываем текущий документ для следующего файла
            current_doc = None

    # Сохраняем результат
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(documents, out_f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Спарсено {len(documents)} документов.")
    print(f"Результат сохранен в {output_path}")


if __name__ == "__main__":
    parse_conllu_to_json(INPUT_FILES, OUTPUT_JSON)
