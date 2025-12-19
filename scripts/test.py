import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device)

#cache_dir = "~/.cache/huggingface/hub"
print("loading gpt2...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

print("loading sae...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device
)

print("done")