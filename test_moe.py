import torch
from torch.profiler import profile, ProfilerActivity, record_function
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts

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

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
                      use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)
    f1,f2, m1,m2 = get_stats(top_k)
    times = get_times("fused_moe_kernel", prof)
    print(f"moe up {(f1/1e6)/(times[0]):.2f} TFLOPs, {(m1/1e3)/times[0]} GB/s")
    print(f"moe down {(f2/1e6)/(times[1]):.2f} TFLOPs, {(m2/1e3)/times[1]} GB/s")
    print(f1, f2, m1, m2)

# Uniform
    print("benchmarking uniform")
    topk_ids = (torch.arange(top_k*num_tokens)%n_experts).reshape(num_tokens, top_k).to(torch.int32)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
                      use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)

    f1,f2, m1,m2 = get_stats(min(top_k*num_tokens, n_experts))
    times = get_times("fused_moe_kernel", prof)
    print(f"moe up {(f1/1e6)/(times[0]):.2f} TFLOPs, {(m1/1e3)/times[0]} GB/s")
    print(f"moe down {(f2/1e6)/(times[1]):.2f} TFLOPs, {(m2/1e3)/times[1]} GB/s")

# Random
    print("benchmarking random")
    topk_ids = torch.randint(low=0, size=(num_tokens, top_k), high=n_experts).to(torch.int32)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
                      use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)

    f1,f2, m1,m2 = get_stats(len(set(topk_ids.flatten().tolist())))
    times = get_times("fused_moe_kernel", prof)
    print(f"moe up {(f1/1e6)/(times[0]):.2f} TFLOPs, {(m1/1e3)/times[0]} GB/s")
    print(f"moe down {(f2/1e6)/(times[1]):.2f} TFLOPs, {(m2/1e3)/times[1]} GB/s")
    print()
    print()
