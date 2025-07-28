import sys
import torch
from torch.utils.cpp_extension import load
from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8, sglang_per_token_group_quant_fp8
from sglang.srt.layers.layernorm import RMSNorm

torch.set_default_device("cuda")
cu_ext = load(name='my_ext', sources=["./my_kernels/interface.cpp",
                                      "./my_kernels/kernels.cu",
                                      "./my_kernels/kernels_fused.cu", ],
              extra_include_paths=["/usr/local/lib/python3.10/dist-packages/flashinfer/data/include/"],
              verbose=True, extra_cuda_cflags=[f"-lineinfo", "--use_fast_math", "-O3"])


# import torch.distributed as dist
# dist.init_process_group(backend="nccl")
# local_rank = dist.get_rank() % torch.cuda.device_count()
# torch.cuda.set_device(local_rank)
#
# from sglang.srt.distributed.parallel_state import set_mscclpp_all_reduce
# set_mscclpp_all_reduce(True)
# from msamp.operators.dist_op import DistOp
# from msamp.common.dtype import Dtypes

H = 7168

for N in [8, 64, 256, 1024, 2048]:
    inp1 = torch.randn((N, H), dtype=torch.bfloat16)
    inp2 = torch.randn((N, H), dtype=torch.bfloat16)

    rms_n = RMSNorm(7168).to(torch.bfloat16)
    inp3 = rms_n.forward_cuda(inp1)
    cu_ext.rms_norm(inp1, inp2, rms_n.weight, 1e-6)


    # q1, s1 = per_token_group_quant_fp8(inp3, 128)
    q1, s1 = sglang_per_token_group_quant_fp8(inp3, 128)
    q2 = torch.empty_like(q1)
    s2 = torch.empty_like(s1)

    cu_ext.rms_norm_quant(inp1, q2, s2, rms_n.weight, 1e-10)
# q2, s2 = per_token_group_quant_fp8(inp2, 128)


    if "debug" in sys.argv:
        # print(inp3[0])
        print("")
        print(q1)
        print(q2)
        # print(q3)
        print("")
        print(s1)
        print(s2)

for N in [8, 64, 256, 1024, 2048]:
    inp1 = torch.randn((N, H), dtype=torch.bfloat16)
    inp2 = torch.randn((N, H), dtype=torch.bfloat16)
    inp3 = inp1.clone()
    inp4 = inp2.clone()

    rms_n = RMSNorm(7168).to(torch.bfloat16)
    out, res = rms_n.forward_cuda(inp1, inp2)
    # cu_ext.rms_norm(inp1, inp2, rms_n.weight, 1e-6)


    # q1, s1 = per_token_group_quant_fp8(inp3, 128)
    q1, s1 = sglang_per_token_group_quant_fp8(out, 128)
    q2 = torch.empty_like(q1)
    s2 = torch.empty_like(s1)

    cu_ext.rms_norm_quant_add(inp3, inp4, q2, s2, rms_n.weight, 1e-10)
# q2, s2 = per_token_group_quant_fp8(inp2, 128)


    if "debug" in sys.argv:
        # print(inp3[0])
        print("")
        print(q1)
        print(q2)
        # print(q3)
        print("")
        print(s1)
        print(s2)
        # print(s3)
# print(torch.max(torch.abs(inp3[0, 0:128])))
#
# print(q1.shape)
# print(s1.shape)
