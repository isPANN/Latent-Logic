import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from transformer_lens import HookedTransformer

LABELS = ["increase", "decrease", "remain constant"]

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@torch.no_grad()
def classify_three_way(model: HookedTransformer, prompts: list[str], batch_size: int):
    """
    Compute P(label | prompt + '\\nAnswer:') using next-token probabilities.
    We score by the FIRST token of each label (simple + fast).
    """
    label_strs = [" increase", " decrease", " remain constant"]  # leading space is important for GPT2 tokenizer
    label_tok = [model.to_tokens(s, prepend_bos=False)[0, 0].item() for s in label_strs]

    choices, probs_out = [], []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        batch_inp = [p.strip() + "\nAnswer:" for p in batch]
        toks = model.to_tokens(batch_inp)

        logits = model(toks)          # [B, S, vocab]
        last_logits = logits[:, -1, :]  # next token distribution at end of prompt
        p = torch.softmax(last_logits, dim=-1)

        p3 = torch.stack([p[:, t] for t in label_tok], dim=1)  # [B, 3]
        p3 = p3 / (p3.sum(dim=1, keepdim=True) + 1e-12)        # renormalize over 3 options

        idx = torch.argmax(p3, dim=1).tolist()
        for i, j in enumerate(idx):
            choices.append(LABELS[j])
            probs_out.append(p3[i].cpu().tolist())

    return choices, probs_out

@torch.no_grad()
def free_generate(model: HookedTransformer, prompts: list[str], max_new_tokens: int, batch_size: int):
    outs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        batch_inp = [p.strip() + "\nAnswer:" for p in batch]

        toks = model.to_tokens(batch_inp)
        prompt_len = toks.shape[1]

        gen = model.generate(
            toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            stop_at_eos=True,
        )

        for i in range(gen.shape[0]):
            # Only decode newly generated tokens
            new_tokens = gen[i, prompt_len:]
            text = model.to_string(new_tokens).strip()
            outs.append(text)

    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="gpt2-small")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--with_free_text", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    args = ap.parse_args()

    device = args.device or pick_device()

    # load data
    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if args.limit and len(records) >= args.limit:
                break

    prompts = [r["prompt"] for r in records]
    labels = [r["label"] for r in records]
    is_cf = [bool(r["is_counterfactual"]) for r in records]
    domain = [r.get("domain", "") for r in records]

    # load model (no dtype args)
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    # 3-way classification
    choice, probs = classify_three_way(model, prompts, batch_size=args.batch_size)

    out = pd.DataFrame({
        "prompt": prompts,
        "label": labels,
        "is_counterfactual": is_cf,
        "domain": domain,
        "model_choice": choice,
        "p_increase": [p[0] for p in probs],
        "p_decrease": [p[1] for p in probs],
        "p_constant": [p[2] for p in probs],
    })

    if args.with_free_text:
        out["free_text"] = free_generate(
            model, prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
        )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[OK] Wrote {args.output_csv}  rows={len(out)}")

if __name__ == "__main__":
    main()