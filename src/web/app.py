import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "model"))

from src.model.model import Kyivan, KyivanConfig
from src.data_pipeline.normalization import normalize_historical_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Kyivan Web")
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
char_vocab = None
id_to_char = None
MASK_TOKEN_ID = None

# Constants
CHECKPOINT_DIR = "C:/Programming/kyivan/checkpoints/checkpoints/checkpoint-2700"
VOCAB_PATH = "C:/Programming/kyivan/prepared_datasets/tokenizer/char_vocab.json"
REGION_NAMES = {
    0: "Новгородский (NW)",
    1: "Юго-Западный (SW)",
    2: "Древневосточнославянский (OES)",
    3: "Церковнославянский (CS)",
}


def bin_to_period(bin_idx: int) -> str:
    start = 800 + bin_idx * 50
    return f"{start}–{start + 50} гг."


class AnalyzeRequest(BaseModel):
    text: str
    temperature: float = 1.0


def load_resources():
    global model, char_vocab, id_to_char, MASK_TOKEN_ID, UNK_MASK_TOKEN_ID

    log.info(f"Loading vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        char_vocab = json.load(f)
    id_to_char = {v: k for k, v in char_vocab.items()}
    MASK_TOKEN_ID = char_vocab.get("[-]")
    UNK_MASK_TOKEN_ID = char_vocab.get("[#]")

    log.info(f"Loading model from {CHECKPOINT_DIR} on {device}...")
    config = KyivanConfig.from_pretrained(CHECKPOINT_DIR)
    config._attn_implementation = "eager"
    config.output_attentions = True
    model = Kyivan(config, num_date_bins=20, num_regions=4)

    from safetensors.torch import load_file

    tensors = load_file(Path(CHECKPOINT_DIR) / "model.safetensors")
    model.load_state_dict(tensors, strict=False)
    model.to(device)
    model.eval()
    log.info("Model loaded successfully!")


load_resources()


@app.get("/")
def read_root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    # 1. Normalize and Tokenize input
    # Protect special characters used by the web interface before normalization
    text = req.text.replace("?", "[[QMARK]]").replace("#", "[[HASH]]")
    text = normalize_historical_text(text)
    text = text.replace("[[QMARK]]", "?").replace("[[HASH]]", "#")

    tokens = ["[SOS]"]
    for char in text:
        if char == "?":
            tokens.append("[-]")
        elif char == "#":
            tokens.append("[#]")
        else:
            tokens.append(char)

    input_ids = [char_vocab.get(t, char_vocab.get("[UNK]")) for t in tokens]
    # The checkpoint model might have a smaller vocab_size than the current char_vocab
    input_ids = [tid if tid < model.config.vocab_size else char_vocab.get("[UNK]") for tid in input_ids]
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, device=device)

    # 2. Run inference
    with torch.no_grad():
        outputs = model(
            input_tensor, attention_mask=attention_mask, output_attentions=True
        )

    logits_restore = outputs.logits_restore[0]  # (seq_len, vocab_size)
    logits_unk = outputs.logits_unk[0]  # (seq_len, 2)
    logits_date = outputs.logits_date[0]  # (num_date_bins)
    logits_region = outputs.logits_region[0]  # (num_regions)

    # Extract attention weights (last layer, averaged across heads)
    last_layer_attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=0)  # (seq_len, seq_len)

    # 3. Process Date & Region
    date_probs = F.softmax(logits_date, dim=0).tolist()
    region_probs = F.softmax(logits_region, dim=0).tolist()

    # The [SOS] token (index 0) makes the date/region prediction.
    sos_attention = avg_attn[0].tolist()

    # 4. Process Restorations
    restorations = []
    for idx, token_id in enumerate(input_ids):
        if token_id == MASK_TOKEN_ID:
            t = max(0.01, req.temperature)  # prevent division by zero
            probs = F.softmax(logits_restore[idx] / t, dim=0)
            topk_probs, topk_indices = torch.topk(probs, 5)

            top_k_list = []
            for p, char_id in zip(topk_probs, topk_indices):
                top_k_list.append(
                    {"char": id_to_char.get(char_id.item(), "?"), "prob": p.item()}
                )

            restorations.append(
                {
                    "token_index": idx,
                    "is_unk": False,
                    "top_k": top_k_list,
                    "attention": avg_attn[idx].tolist(),
                }
            )
        elif token_id == UNK_MASK_TOKEN_ID:
            probs = F.softmax(logits_unk[idx], dim=0)
            
            restorations.append(
                {
                    "token_index": idx,
                    "is_unk": True,
                    "prob_multi": probs[1].item(),
                    "prob_single": probs[0].item(),
                    "attention": avg_attn[idx].tolist(),
                }
            )

    # Normalize SOS attention for frontend (min-max scaling or just raw values to let frontend handle)
    # Actually, raw values sum to 1.0 because of softmax in attention.

    return {
        "tokens": tokens,
        "date_probs": [
            {"period": bin_to_period(i), "prob": p} for i, p in enumerate(date_probs)
        ],
        "region_probs": [
            {"region": REGION_NAMES.get(i, f"Region {i}"), "prob": p}
            for i, p in enumerate(region_probs)
        ],
        "sos_attention": sos_attention,
        "restorations": restorations,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
