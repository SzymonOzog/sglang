import torch
import sys
import argparse
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
from triton.testing import do_bench
torch.utils.cpp_extension.COMMON_NVCC_FLAGS = []

def interleave_tensor(tensor):
    """
    Interleave a tensor of shape (M, 256, K) by alternating chunks of 8
    from the first half (0-127) and second half (128-255) of dimension 1.
    Args:
        tensor: PyTorch tensor of shape (M, 256, K)
    Returns:
        Interleaved tensor of shape (M, 256, K)
    """
    M, _, K = tensor.shape

    first_half = tensor[:, :128, :]
    second_half = tensor[:, 128:, :]

    first_chunks = first_half.view(M, 16, 8, K)
    second_chunks = second_half.view(M, 16, 8, K)

    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, 256, K)

    return result.contiguous()

KERNEL_VARIANTS=13
my_ext = load(name="my_ext", sources = ["./csrc/torch_interface.cpp",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_prefetching.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_smem.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_db.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_tb.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_mb.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_sacc.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_pc.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_ast.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_tma.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_swiglu.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_tma_swiglu.cu",
                                        "./csrc/kernels/fused_moe_w8a8/fused_mow_w8a8_up_down.cu",
                                        ], extra_cuda_cflags=["-lineinfo"])


def bench_events(fn, num_warmups: int = 5, num_tests: int = 50,
          high_precision: bool = False):
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        y = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        x @ y

    total_time = 0
    # Testing
    for i in range(num_tests):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Flush L2 cache with 256 MB data
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
        cache.zero_()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()

        total_time += start_event.elapsed_time(end_event)
    return total_time*1e3 / num_tests

# Adapted from: https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/utils.py
def bench_kineto(fn, kernel_name: str = "moe", num_tests: int = 50):
    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for i in range(2):
            for _ in range(num_tests):
                # Alocate big tensor to clear cache
                x = torch.randn((8, 8192, 8192), dtype=torch.float, device='cuda')
                fn()
                torch.cuda.synchronize()
            prof.step()

    # Parse the profiling table
    times = []
    for e in prof.profiler.function_events:
        if kernel_name in e.name:
            times.append(e.device_time_total)
    return sum(times)/len(times)

def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    """
    Generate topk_ids with a given balancedness.

    balancedness:
        1.0 -> perfectly balanced (uniform)
        0.0 -> maximally skewed (all tokens go to one expert)
        in between -> mixture
    """
    # interpolate between uniform and skewed distribution
    uniform = torch.ones(num_experts) / num_experts
    skewed = torch.zeros(num_experts); skewed[0] = 1.0
    probs = balancedness * uniform + (1 - balancedness) * skewed

    # sample expert assignments
    topk_ids = torch.multinomial(probs, num_tokens * top_k, replacement=True)
    topk_ids = topk_ids.view(num_tokens, top_k)
    return topk_ids

def get_stats(activated_experts):
    flops_1 = 2*num_tokens*w1.shape[1]*w1.shape[2]
    flops_2 = 2*num_tokens*top_k*w2.shape[1]*w2.shape[2]

    mem_1 = activated_experts*w1.shape[1]*w1.shape[2] * w1.element_size() + \
            activated_experts*w1.shape[1]//block_shape[0]*w1.shape[2]//block_shape[0] * w1_scale.element_size() + \
            hidden_size*num_tokens + num_tokens*hidden_size//block_shape[0] * 4 + \
            top_k * num_tokens * w1.shape[2] * 2

    mem_2 = activated_experts*w2.shape[1]*w2.shape[2] * w2.element_size() + \
            activated_experts*w2.shape[1]//block_shape[0]*w2.shape[2]//block_shape[0] * w2_scale.element_size() + \
            hidden_size*num_tokens + activated_experts*num_tokens*hidden_size//block_shape[0] * 4 + \
            num_tokens * w2.shape[2] * 2
    return flops_1, flops_2, mem_1, mem_2

def get_times(kernel_name, prof):
    ret = []
    for e in prof.profiler.function_events:
        if kernel_name in e.name:
            ret.append(e.device_time_total)
    return ret


def run_moe(topk_ids, eps=1e-10):
    x = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16).normal_(mean=0, std=0.55)
    x_q, x_scale = sglang_per_token_group_quant_fp8(x, block_shape[1])
    x_sc = x_scale.repeat_interleave(block_shape[0], 1)

    x_dq = x_q.to(torch.bfloat16) * x_sc
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], n_experts)

    out_triton_up = torch.empty((num_tokens, top_k, w1.shape[1]), device=x.device, dtype=x.dtype)
    out_triton_swiglu = torch.empty((num_tokens*top_k, w1.shape[1]//2), device=x.device, dtype=x.dtype)
    out_custom_swiglu = out_triton_swiglu.clone()
    out_triton_down = torch.empty((num_tokens, top_k, x.shape[1]), device=x.device, dtype=x.dtype)
    out_triton = torch.empty_like(x)

    compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    bench_fn = bench_kineto
    triton_time_up = float("inf")
    triton_time_down = float("inf")
    # warmup triton compilation
    if profiling:
        triton_time_up = bench_fn(lambda: invoke_fused_moe_kernel(x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                                                                  sorted_token_ids, expert_ids, num_tokens_post_padded,
                                                                  False, top_k, config, compute_type, True, False, False, False, False, block_shape))
        triton_time_swiglu = bench_fn(lambda: silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu),
                                      kernel_name="silu")
        triton_time_up += triton_time_swiglu
        triton_time_down = bench_fn(lambda: invoke_fused_moe_kernel(out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                                                                     sorted_token_ids, expert_ids, num_tokens_post_padded,
                                                                     False, 1, config, compute_type, True, False, False, False, False, block_shape))

    if profiling:
        triton_time_up = bench_fn(lambda: invoke_fused_moe_kernel(x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                                                                  sorted_token_ids, expert_ids, num_tokens_post_padded,
                                                                  False, top_k, config, compute_type, True, False, False, False, False, block_shape))
        triton_time_swiglu = bench_fn(lambda: silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu),
                                      kernel_name="silu")
        triton_time_up += triton_time_swiglu
        triton_time_down = bench_fn(lambda: invoke_fused_moe_kernel(out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                                                                     sorted_token_ids, expert_ids, num_tokens_post_padded,
                                                                     False, 1, config, compute_type, True, False, False, False, False, block_shape))

    invoke_fused_moe_kernel(x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                            sorted_token_ids, expert_ids, num_tokens_post_padded,
                            False, top_k, config, compute_type, True, False, False, False, False, block_shape)
    silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu)
    invoke_fused_moe_kernel(out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                            sorted_token_ids, expert_ids, num_tokens_post_padded,
                            True, 1, config, compute_type, True, False, False, False, False, block_shape)
    moe_sum_reduce_torch_compile(out_triton_down.view(*out_triton_down.shape), out_triton, moe_config.routed_scaling_factor)

    # Sanity check that we implemented it all correctly
    # out_layer = fused_experts(x, w1, w2, (topk_weights, topk_ids, None), moe_config,
    #               use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, block_shape=block_shape)
    # # close to 0 values can have high rtol
    # assert(torch.allclose(out_triton, out_layer, rtol=rtol))

    # print(sorted_token_ids[128*16:num_tokens_post_padded[0]])
    # print(expert_ids)
    # out = my_ext.fused_moe_w8a8(x_q, x_scale, w1, w1_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, 0)
    best_configuration_up = ""
    best_configuration_down = ""
    best_diff_up = (-1, -1)
    best_diff_down = (-1, -1)
    best_diff_swiglu = (-1, -1)
    best_up_time = float("inf")
    best_down_time = float("inf")
    variants = [variant] if variant else list(range(KERNEL_VARIANTS))
    block_m = 16
    for kernel_variant in variants:
        for block_m in range(8, 52, 8) if kernel_variant > 8 else [16]:
            if num_tokens < block_m and block_m != 16:
                continue
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, n_experts)
            configuration = f"{kernel_variant=} {block_m=}"
            out = my_ext.fused_moe_w8a8(x_q, x_scale, w1_swiglu, w1_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, kernel_variant, block_m)

            if kernel_variant < 11:
                idx = torch.isclose(out, out_triton_up.reshape(out.shape), atol=atol, rtol=rtol).logical_not()
                # if not torch.allclose(out, out_triton_up.reshape(out.shape), atol=atol, rtol=rtol):
                #     # print(idx)
                    # print(idx.nonzero())
                    # print(idx.sum()/out.nelement())
                    # print(out_triton_up.reshape(out.shape)[idx][:10])
                    # print(out[idx][:10])
                #     # print()
                    # st = 64
                    # print(out_triton_up.reshape(out.shape)[0][st:st +32])
                    # print(out[0][st:st +32])
                #     for i in range(0, 64, 4):
                #         s = ", ".join(f'{tk:.2f}'  for tk in out[0, i:i+4].tolist())
                #         s2 = ", ".join(f'{tk:.2f}' for tk in out_triton_up.reshape(out.shape)[0, i:i+4].tolist())
                #         print(i, s, "   ", s2)
                # return

                # torch.cuda.synchronize()
                # print("x")
                # for i in range(0, hidden_size, 4):
                #     print(i, ", ".join(f'{tk:.5f}' for tk in x_dq[0, i:i+4].tolist()))
                # print("w")
                # for i in range(0, w1.shape[-1], 4):
                #     print(i, ", ".join(f'{tk:.5f}' for tk in w1[0,16 ,i:i+4].tolist()))
                # for i in range(0, hidden_size, 4):
                #     print(i, ", ".join(f'{tk:.5f}' for tk in w1_dq[0,16 ,i:i+4].tolist()))
                # return
                # print(w1_dq[0])
                # print(w1[0])
                # print(w1_scale[0])
                # print(w1_scale.shape)
                # print(w1.stride())
                # print(out.shape)
                # print(x_scale[0])
                # return
                # print(w1.shape)

                # assert(torch.allclose(out, out_triton_up.reshape(out.shape), atol=atol, rtol=rtol))
                diff = torch.abs(out-out_triton_up.reshape(out.shape))
                mean_diff_up = diff.mean()
                max_diff_up = diff.max()

                silu_and_mul(out, out_custom_swiglu)
            else:
                out_custom_swiglu = out.clone()
                mean_diff_up = 0
                max_diff_up = 0
            # idx = torch.isclose(out_custom_swiglu, out_triton_swiglu, atol=atol, rtol=rtol).logical_not()
            # if not torch.allclose(out_custom_swiglu, out_triton_swiglu, atol=atol, rtol=rtol):
            #     # print(idx)
            #     print(idx.nonzero())
            #     print(idx.sum()/out_triton_swiglu.nelement())
            #     print(out_triton_swiglu[idx][:10])
            #     print(out_custom_swiglu[idx][:10])
            #     print(out_triton_swiglu[:10])
            #     print(out_custom_swiglu[:10])
            #     print(out_triton_up[0,0, 32:32+10])
            #     print(out_triton_up[0,0, 128+32:128+32+10])

            # assert(torch.allclose(out_custom_swiglu, out_triton_swiglu, atol=atol, rtol=rtol))
            diff = torch.abs(out_custom_swiglu - out_triton_swiglu)
            mean_diff_swiglu = diff.mean()
            max_diff_swiglu = diff.max()

            out_custom_swiglu = out_triton_swiglu.clone()
            s_q, s_scale = sglang_per_token_group_quant_fp8(out_custom_swiglu, block_shape[1])
            # out = my_ext.fused_moe_w8a8(x_q, x_scale, w2, w2_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, 1, 0)
            out = my_ext.fused_moe_w8a8(s_q, s_scale, w2, w2_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, 1, min(kernel_variant, 10), block_m)
            out *= topk_weights.view((num_tokens*top_k, 1))

            # idx = torch.isclose(out, out_triton_down.reshape(out.shape), atol=atol, rtol=rtol).logical_not()
            # if not torch.allclose(out, out_triton_down.reshape(out.shape), atol=atol, rtol=rtol):
            #     # print(idx.nonzero())
            #     print(idx.sum()/out.nelement())
            #     print(out_triton_down.reshape(out.shape)[idx][:10])
            #     print(out[idx][:10])
            # return

            # TODO swiglu too big stacks too much error
            # assert(torch.allclose(out, out_triton_down.reshape(out.shape), atol=10*atol, rtol=rtol))
            diff = torch.abs(out-out_triton_down.reshape(out.shape))
            mean_diff_down = diff.mean()
            max_diff_down = diff.max()
            if profiling:
                if kernel_variant > 10:
                    up_time = bench_fn(lambda: my_ext.fused_moe_w8a8(x_q, x_scale, w1, w1_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, kernel_variant, block_m))
                    if up_time < best_up_time:
                        best_up_time = up_time
                        best_diff_up = (mean_diff_up, max_diff_up)
                        best_diff_swiglu = (mean_diff_swiglu, max_diff_swiglu)
                        best_configuration_up = configuration
                    else:
                        down_time = bench_fn(lambda: my_ext.fused_moe_w8a8(s_q, s_scale, w2, w2_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, 1, kernel_variant, block_m))
                        if down_time < best_down_time:
                            best_down_time = down_time
                            best_diff_down = (mean_diff_down, max_diff_down)
                            best_configuration_down = configuration
                if verbose:
                    print(f"{configuration=}, {up_time=:.2f} us, {down_time=:.2f} us")
    return [*best_diff_up, *best_diff_swiglu, *best_diff_down], [best_up_time, best_down_time], [triton_time_up, triton_time_down], best_configuration_up, best_configuration_down

def parse_arguments():
    parser = argparse.ArgumentParser(description='MOE Benchmark Script')

    parser.add_argument('--batch-sizes',
                        type=int,
                        nargs='+',
                        default=[8, 32, 128, 256],
                        help='Batch sizes to benchmark (default: [8, 32, 128, 256])')

    parser.add_argument('--profile',
                        action='store_true',
                        help='Enable profiling mode')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose mode')

    parser.add_argument('--variant',
                        type=int,
                        default=None,
                        help='What variant of kernel to profile')

    parser.add_argument('--balancedness',
                        type=float,
                        nargs='+',
                        default=[],
                        help='Balancedness values to test (default: [])')

    return parser.parse_args()

torch.manual_seed(42)
(w1, w2, w1_scale, w2_scale, moe_config) = torch.load("./moe_config.pt", weights_only=False)
w1 = w1.to("cuda:0")
w2 = w2.to("cuda:0")
w1_scale = w1_scale.to("cuda:0")
w2_scale = w2_scale.to("cuda:0")
torch.set_default_device("cuda:0")

num_tokens = 8192
# hidden_size = 128
hidden_size = 7168
top_k = 9 # 8 picked + 1 shared
block_shape = [128, 128]
n_experts = 257
atol = 1e-3
rtol = 1e-2

w1_scale = torch.randn((n_experts, w1.shape[1]//block_shape[0], w1.shape[2]//block_shape[1]), dtype=w2_scale.dtype) * 0.01
w1 = w1[..., :hidden_size].contiguous()
w1_swiglu = interleave_tensor(w1)
w1_scale = w1_scale[..., :hidden_size//128].reshape(257, 2, hidden_size//128).contiguous()
w1_dq = w1.to(torch.bfloat16) * w1_scale.repeat_interleave(block_shape[0], 2).repeat_interleave(block_shape[0], 1)
moe_config.inplace=False

def bench():
    diffs, cu_times, t_times, configuration_up, configuration_down = run_moe(topk_ids)
    # t_times = get_times("fused_moe_kernel", prof)
    # cu_times = get_times("fused_moe_w8a8", prof)

    f1,f2, m1,m2 = get_stats(len(set(topk_ids.flatten().tolist())))
    print(f"Triton moe up {t_times[0]:.2f} us {(f1/1e6)/(t_times[0]):.2f} TFLOPs, {(m1/1e3)/t_times[0]:.2f} GB/s")
    print(f"AA moe up{configuration_up=} {cu_times[0]:.2f} us {(f1/1e6)/(cu_times[0]):.2f} TFLOPs, {(m1/1e3)/cu_times[0]:.2f} GB/s, speed relative to triton {t_times[0]*100/cu_times[0]:.2f}%")

    print(f"Triton moe down {cu_times[1]:.2f} us {(f2/1e6)/(t_times[1]):.2f} TFLOPs, {(m2/1e3)/t_times[1]:.2f} GB/s")
    print(f"AA moe {configuration_down=} {cu_times[1]:.2f} us down {(f2/1e6)/(cu_times[1]):.2f} TFLOPs, {(m2/1e3)/cu_times[1]:.2f} GB/s, speed relative to triton {t_times[1]*100/cu_times[1]:.2f}%")

    print(f"""Up projection mean abs difference {diffs[0]:.2f},
Up projection max abs difference {diffs[1]:.2f},
swiglu mean abs difference {diffs[2]:.2f},
swiglu max abs difference {diffs[3]:.2f},
down projection mean abs difference {diffs[4]:.2f},
down projection max abs difference {diffs[5]:.2f},""")
    print("")

if __name__ == "__main__":
    args = parse_arguments()

    profiling = args.profile
    batch_sizes = args.batch_sizes
    balancedness_values = args.balancedness
    variant = args.variant
    verbose = args.verbose

    for num_tokens in batch_sizes:
        print("Batch size", num_tokens)
        config_dtype = 'fp8_w8a8'
        config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, config_dtype, block_shape=block_shape, M=num_tokens)
        print(config)
        topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.bfloat16), dim=-1)

        # Uniform
        print("benchmarking uniform")
        topk_ids = (torch.arange((top_k-1)*num_tokens)%n_experts).reshape(num_tokens, top_k-1).to(torch.int32)
        # add shared expert to every token
        topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(n_experts-1)))
        if profiling:
            bench()
        else:
            run_moe(topk_ids)

        # Varying balancedness
        for balancedness in balancedness_values:
            print(f"benchmarking {balancedness=}")
            topk_ids = generate_topk_ids(n_experts-1, num_tokens, top_k-1)
            # add shared expert to every token
            topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(n_experts-1)))
            if profiling:
                bench()
            else:
                run_moe(topk_ids)
