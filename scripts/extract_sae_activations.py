# extract_sae_acts.py
import os
import json
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import SAE


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def extract_topk_sae_acts(
    model: HookedTransformer,
    sae: SAE,
    prompts: list[str],
    hook_name: str,
    topk: int,
    batch_size: int,
):
    """
    Returns sparse COO rows:
      [sample_id, feature_id, act_value]
    computed at the final token position of: prompt + "\\nAnswer:"
    """
    device = model.cfg.device
    rows = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        # Standardize the decoding position: right before the model would generate the first answer token
        batch_inputs = [p.strip() + "\nAnswer:" for p in batch_prompts]

        tokens = model.to_tokens(batch_inputs)  # [B, S]
        # "Answer:" is at the end; we want the last non-pad token index for each row
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = model.tokenizer.eos_token_id

        # last non-pad position
        nonpad = (tokens != pad_id).int()
        lengths = nonpad.sum(dim=1)  # [B]
        pos = lengths - 1            # [B]

        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name]     # [B, S, d_model]

        b_idx = torch.arange(resid.shape[0], device=resid.device)
        resid_at = resid[b_idx, pos, :]          # [B, d_model]
        acts = sae.encode(resid_at)              # [B, d_sae]

        k = min(topk, acts.shape[1])
        topk_idx = torch.topk(acts, k=k, dim=1).indices  # [B, k]

        # Gather values
        topk_vals = torch.gather(acts, 1, topk_idx)      # [B, k]

        # Move small tensors to CPU for row building
        topk_idx = topk_idx.cpu().tolist()
        topk_vals = topk_vals.cpu().tolist()

        for i in range(len(batch_prompts)):
            sample_id = start + i
            for fid, val in zip(topk_idx[i], topk_vals[i]):
                rows.append([sample_id, int(fid), float(val)])

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="gpt2-small")
    ap.add_argument("--sae_release", type=str, default="gpt2-small-res-jb")
    ap.add_argument("--sae_id", type=str, default="blocks.8.hook_resid_pre")
    ap.add_argument("--hook_name", type=str, default="blocks.8.hook_resid_pre")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    device = args.device or pick_device()

    # Load dataset
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

    # Load model + SAE
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )

    # Extract sparse top-k activations
    rows = extract_topk_sae_acts(
        model=model,
        sae=sae,
        prompts=prompts,
        hook_name=args.hook_name,
        topk=args.topk,
        batch_size=args.batch_size,
    )

    # Build output table: sparse activations + metadata per sample
    acts_df = pd.DataFrame(rows, columns=["sample_id", "feature_id", "act_value"])

    meta_df = pd.DataFrame({
        "sample_id": list(range(len(records))),
        "label": labels,
        "is_counterfactual": is_cf,
        "domain": domain,
        "prompt": prompts,  # keep for debugging; you can drop later
    })

    out_df = acts_df.merge(meta_df, on="sample_id", how="left")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    # Sanity checks
    avg_topk = out_df.groupby("sample_id").size().mean()
    print(f"[OK] Wrote {args.output_csv}")
    print(f"[Sanity] samples={len(records)}, avg_nonzero_per_sample={avg_topk:.1f} (expected ~{args.topk})")
    print(f"[Sanity] unique_features={out_df['feature_id'].nunique()}")


if __name__ == "__main__":
    main()