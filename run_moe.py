import torch
from torch.profiler import profile, ProfilerActivity, record_function
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        fused_experts,
        try_get_optimal_moe_config,
        moe_align_block_size,
        invoke_fused_moe_kernel,
        moe_sum_reduce_triton,
        moe_sum_reduce_torch_compile,
        )
from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        )
from torch.utils.cpp_extension import load
from sgl_kernel import gelu_and_mul, silu_and_mul
import triton.language as tl

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
moe_config.inplace=False

for num_tokens in [8, 256, 1024, 8192]:
    print("Batch size", num_tokens)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    x_q, x_scale = sglang_per_token_group_quant_fp8(x, block_shape[1])
    topk_weights = torch.randn((num_tokens, top_k), dtype=torch.bfloat16)

    config_dtype = 'fp8_w8a8'

# Ideal
    print("benchmarking ideal")
    topk_ids = torch.arange(top_k).repeat(num_tokens,1).to(torch.int32)

    config = try_get_optimal_moe_config(w1.shape, w2.shape, topk_ids.shape[1], config_dtype, block_shape=block_shape, M=num_tokens)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], n_experts)

    out_triton_up = torch.empty((num_tokens, top_k, w1.shape[1]), device=x.device, dtype=x.dtype)
    out_triton_swiglu = torch.empty((num_tokens*top_k, w1.shape[1]//2), device=x.device, dtype=x.dtype)
    out_triton_down = torch.empty((num_tokens, top_k, x.shape[1]), device=x.device, dtype=x.dtype)
    out_triton = torch.empty_like(x)

    compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16

    invoke_fused_moe_kernel(x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                            sorted_token_ids, expert_ids, num_tokens_post_padded,
                            False, top_k, config, compute_type, True, False, False, False, False, block_shape)
    print(out_triton_up)
    silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu)
    invoke_fused_moe_kernel(out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                            sorted_token_ids, expert_ids, num_tokens_post_padded,
                            True, 1, config, compute_type, True, False, False, False, False, block_shape)
    moe_sum_reduce_torch_compile(out_triton_down.view(*out_triton_down.shape), out_triton, moe_config.routed_scaling_factor)

    out = my_ext.fused_moe_w8a8(x_q, x_scale, w1, w1_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k)

    # Sanity check that we implemented it all correctly
    out_layer = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
                  use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)
    # close to 0 values can have high rtol
    assert(torch.allclose(out_triton, out_layer, rtol=1e-2))
