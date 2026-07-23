import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.model import Kyivan, KyivanConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def iterative_predict(model, input_ids_list, mask_token_id, device):
    current_ids = list(input_ids_list)
    mask_indices = [i for i, tid in enumerate(current_ids) if tid == mask_token_id]
    
    # Store predictions in a dict: {idx: predicted_char_id}
    predictions = {}
    
    with torch.no_grad():
        while mask_indices:
            t_input = torch.tensor([current_ids], dtype=torch.long, device=device)
            t_mask = torch.ones_like(t_input, device=device)
            outputs = model(t_input, attention_mask=t_mask)
            logits_restore = outputs.logits_restore[0]
            
            best_idx = -1
            best_prob = -1.0
            best_char_id = -1
            
            for idx in mask_indices:
                probs = F.softmax(logits_restore[idx], dim=0)
                top_prob, top_char = torch.max(probs, dim=0)
                
                if top_prob.item() > best_prob:
                    best_prob = top_prob.item()
                    best_idx = idx
                    best_char_id = top_char.item()
            
            # Fill the most confident
            current_ids[best_idx] = best_char_id
            mask_indices.remove(best_idx)
            predictions[best_idx] = best_char_id

    return predictions

def evaluate_file(file_path, model, char_vocab, device):
    log.info(f"Evaluating {file_path} ...")
    mask_token_id = char_vocab["[-]"]
    
    total_masks = 0
    correct_masks = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Processing"):
        data = json.loads(line)
        input_ids = data["input_ids"]
        labels = data["labels_res"]  # Target char ids. -100 means ignore.
        
        # Only evaluate on examples that have masks
        if mask_token_id not in input_ids:
            continue
            
        predictions = iterative_predict(model, input_ids, mask_token_id, device)
        
        for idx, pred_id in predictions.items():
            true_id = labels[idx]
            if true_id != -100:
                total_masks += 1
                if pred_id == true_id:
                    correct_masks += 1
                    
    if total_masks > 0:
        accuracy = correct_masks / total_masks
        log.info(f"Results for {Path(file_path).name}: Accuracy (hit@1) = {accuracy:.4f} ({correct_masks}/{total_masks})")
    else:
        log.info(f"No mask tokens found in {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--vocab", type=str, default="prepared_datasets/tokenizer/char_vocab.json")
    parser.add_argument("--test_a", type=str, default="human_readable_datasets/test_a.jsonl")
    parser.add_argument("--test_b", type=str, default="human_readable_datasets/test_b.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.vocab, "r", encoding="utf-8") as f:
        char_vocab = json.load(f)
        
    log.info(f"Loading model from {args.checkpoint}...")
    config = KyivanConfig.from_pretrained(args.checkpoint)
    config._attn_implementation = "eager"
    model = Kyivan(config, num_date_bins=20, num_regions=4)
    
    tensors = load_file(Path(args.checkpoint) / "model.safetensors")
    model.load_state_dict(tensors, strict=False)
    model.to(device)
    model.eval()
    log.info("Model loaded.")

    if Path(args.test_a).exists():
        evaluate_file(args.test_a, model, char_vocab, device)
    else:
        log.warning(f"{args.test_a} not found.")
        
    if Path(args.test_b).exists():
        evaluate_file(args.test_b, model, char_vocab, device)
    else:
        log.warning(f"{args.test_b} not found.")

if __name__ == "__main__":
    main()
