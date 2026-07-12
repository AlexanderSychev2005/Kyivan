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
        json_path = f"human_readable_datasets/{split_name}.jsonl"
        if os.path.exists(json_path):
            print(f"Uploading readable text for '{split_name}'...")
            api.upload_file(
                path_or_fileobj=json_path,
                path_in_repo=f"readable_texts/{split_name}.jsonl",
                repo_id=args.repo_id,
                repo_type="dataset",
                token=args.token,
            )

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
