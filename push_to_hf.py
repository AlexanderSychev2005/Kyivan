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
    for split_name, split_ds in ds.items():
        # test_b carries pre-computed "labels" (real historical lacunae) while
        # train/eval/test_a don't (masked dynamically by the collator at load
        # time) -- the Hub requires matching features across splits of the
        # same config, so test_b gets its own config.
        config_name = "test_b" if split_name == "test_b" else "default"
        print(f"Pushing split '{split_name}' (config '{config_name}')...")
        split_ds.push_to_hub(
            args.repo_id, config_name=config_name, split=split_name, token=args.token
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

    # 3. Upload label configs
    label_path = "prepared_datasets/label_configs.json"
    if os.path.exists(label_path):
        print(f"Pushing label configs ({label_path})...")
        api.upload_file(
            path_or_fileobj=label_path,
            path_in_repo="configs/label_configs.json",
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
