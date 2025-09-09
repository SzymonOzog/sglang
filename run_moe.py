import torch
from torch.profiler import profile, ProfilerActivity, record_function
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from torch.utils.cpp_extension import load
my_ext = load(name="my_ext", sources = ["interface.cpp", "fused_moe_w8a8.cu"])

def get_stats(activated_experts):
    flops_1 = 2*num_tokens*w1.shape[1]*w1.shape[2]
    flops_2 = 2*num_tokens*top_k*w2.shape[1]*w2.shape[2]

    mem_1 = activated_experts*w1.shape[1]*w1.shape[2] * w1.element_size() + \
            activated_experts*w1.shape[1]//block_shape[0]*w1.shape[2]//block_shape[0] * w1_scale.element_size() + \
            x.numel() + num_tokens*x.shape[1]//block_shape[0] * 4 + \
            top_k * num_tokens * w1.shape[2] * 2

    mem_2 = activated_experts*w2.shape[1]*w2.shape[2] * w2.element_size() + \
            activated_experts*w2.shape[1]//block_shape[0]*w2.shape[2]//block_shape[0] * w2_scale.element_size() + \
            x.numel() + activated_experts*num_tokens*x.shape[1]//block_shape[0] * 4 + \
            num_tokens * w2.shape[2] * 2
    return flops_1, flops_2, mem_1, mem_2

def get_times(kernel_name, prof):
    ret = []
    for e in prof.profiler.function_events:
        if kernel_name == e.name:
            ret.append(e.device_time_total)
    return ret

torch.manual_seed(42)
(w1, w2, w1_scale, w2_scale, moe_config) = torch.load("./python/sglang/moe_config.pt", weights_only=False)
w1 = w1.to("cuda:0")
w2 = w2.to("cuda:0")
w1_scale = w1_scale.to("cuda:0")
w2_scale = w2_scale.to("cuda:0")
torch.set_default_device("cuda:0")
print("")
print(w1.dtype, w1_scale.dtype)

num_tokens = 8192
hidden_size = 7168
top_k = 9 # 8 picked + 1 shared
block_shape = [128, 128]
n_experts = 257

for num_tokens in [8, 256, 1024, 8192]:
    print("Batch size", num_tokens)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    topk_weights = torch.randn((num_tokens, top_k), dtype=torch.bfloat16)

# Ideal
    print("benchmarking ideal")
    topk_ids = torch.arange(top_k).repeat(num_tokens,1).to(torch.int32)

    print(my_ext.fused_moe_w8a8(x, x, w1, w1_scale, topk_ids, topk_ids, topk_ids, top_k).shape)
