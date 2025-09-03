import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts

(w1, w2, w1_scale, w2_scale, moe_config) = torch.load("./python/sglang/moe_config.pt", weights_only=False)
w1 = w1.to("cuda:0")
w2 = w2.to("cuda:0")
w1_scale = w1_scale.to("cuda:0")
w2_scale = w2_scale.to("cuda:0")
torch.set_default_device("cuda:0")
print("")

num_tokens = 8
hidden_size = 7168
top_k = 9 # 8 picked + 1 shared
block_shape = [128, 128]

x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
topk_weights = torch.randn((num_tokens, top_k), dtype=torch.bfloat16)
topk_ids = torch.ones((num_tokens, top_k), dtype=torch.int32)

out = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
              use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)
print(out)

