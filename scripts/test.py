import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# 1. 加载模型 (GPT-2 Small)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. 加载 SAE (选择 Layer 8 的残差流 SAE)
# release_id 和 sae_id 可以在 SAELens 文档或 HuggingFace 查到
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", 
    sae_id = "blocks.8.hook_resid_pre", 
    device = device
)