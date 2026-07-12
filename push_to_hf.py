import argparse
import os
import json
from datasets import load_from_disk
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(
        description="Push Kyivan dataset and tokenizer to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Your Hugging Face repo ID (e.g., username/kyivan-dataset)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face Write Token (optional if already logged in via huggingface-cli)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="prepared_datasets/hf_dataset",
        help="Path to the local HF dataset",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="prepared_datasets/tokenizer/char_vocab.json",
        help="Path to the character vocabulary file",
    )

    args = parser.parse_args()

    # 1. Push the dataset
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)

    api = HfApi()
    
    print(f"Pushing dataset to https://huggingface.co/datasets/{args.repo_id} ...")
    # Pushing dataset splits individually since test_b has 'labels' feature which the others don't
    for split_name, split_ds in ds.items():
        print(f"Pushing split '{split_name}'...")
        config_name = "test_b" if split_name == "test_b" else "default"
        split_ds.push_to_hub(
            args.repo_id, config_name=config_name, split=split_name, token=args.token
        )
        
        print(f"Generating readable JSON for '{split_name}'...")
        
        out_dir = "human_readable_datasets"
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{split_name}.jsonl")
        
        with open(json_path, "w", encoding="utf-8") as f:
            for item in split_ds:
                meta = json.loads(item["metadata"]) if "metadata" in item else {}
                
                # Extract nice date
                date_str = "Unknown"
                if "date_interval" in meta and meta["date_interval"]:
                    date_str = f"{meta['date_interval'][0]} - {meta['date_interval'][1]}"
                elif "date_number" in meta and meta["date_number"]:
                    date_str = str(meta["date_number"])
                
                human_item = {
                    "text": item.get("original_text", ""),
                    "dialect": meta.get("macro_dialect", "Unknown"),
                    "date": date_str,
                    "doc_id": meta.get("doc_id", "Unknown"),
                    "category": meta.get("category", "Unknown")
                }
                
                if "text_with_missing" in item:
                    human_item["text_with_missing"] = item["text_with_missing"]
                    
                f.write(json.dumps(human_item, ensure_ascii=False) + "\n")
        
        api.upload_file(
            path_or_fileobj=json_path,
            path_in_repo=f"readable_texts/{split_name}.jsonl",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token,
        )
        print(f"Uploaded readable texts for '{split_name}' to readable_texts/")

    # 2. Upload the tokenizer file to the same repository
    print(f"Pushing vocabulary file ({args.vocab_path}) to the same repo...")
    api.upload_file(
        path_or_fileobj=args.vocab_path,
        path_in_repo="tokenizer/char_vocab.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
    )

    print("\n✅ Successfully pushed dataset and tokenizer to Hugging Face Hub! 🎉")
    print(
        f"You can view your dataset here: https://huggingface.co/datasets/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
