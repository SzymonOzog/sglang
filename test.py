import sys
import torch
from torch.utils.cpp_extension import load
from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8, sglang_per_token_group_quant_fp8
from sglang.srt.layers.layernorm import RMSNorm
from sgl_kernel import fused_rmsnorm_quant

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
# failed = torch.load("failed.pt")
# t = failed[0].to("cuda:0")
# state = failed[1]
# state["weight"].to("cuda:0")
# print(t)
# print(state["weight"].to("cuda:0"))
# rms_n = RMSNorm(t.shape[-1]).to(torch.bfloat16)
# rms_n.load_state_dict(state)
# inp3 = rms_n.forward_cuda(t)
# # q1, s1 = per_token_group_quant_fp8(inp3, 128, column_major_scales=True, scale_tma_aligned=True)
# q1, s1 = sglang_per_token_group_quant_fp8(inp3.view(-1, inp3.shape[-1]), 128, column_major_scales=True, scale_tma_aligned=True)
# q2 = torch.empty_like(q1)
# s2 = torch.zeros_like(s1)
#
# acc = torch.sum(t * t, dim=1)/t.shape[1]
# srt = torch.rsqrt(acc+1e-6)
# # print("acc", acc)
# # print("rsqrt", srt)
# cu_ext.rms_norm_quant(t, q2, s2, rms_n.weight, 1e-6)
# # print(rms_n.weight.dtype)
# # print(q1)
# # print(q2)
#
# print(s1)
# print(s2)
# # print(torch.max(inp3))
# # print(torch.max(t))
# # print(t.shape)
#
# # print(q1.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()][:10])
# # print(q2.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()][:10])
# dq = s1.repeat_interleave(128).reshape((q1.shape))
# out2 = (q1.to(torch.bfloat16)*dq).to(torch.bfloat16)
#
# dq = s2.repeat_interleave(128).reshape((q1.shape))
# out3 = (q2.to(torch.bfloat16)*dq).to(torch.bfloat16)
# print(out2[:10])
# print(out3[:10])
# print(t[:10])
# print(inp3[:10])
#
for N in [8, 64, 256, 1024, 2048]:
    inp1 = torch.randn((N, H), dtype=torch.bfloat16)
    inp2 = torch.randn((N, H), dtype=torch.bfloat16)

    rms_n = RMSNorm(7168).to(torch.bfloat16)
    inp3 = rms_n.forward_cuda(inp1)

    # print(isinstance(inp3, tuple))
    # q1, s1 = inp3
    cu_ext.rms_norm(inp1, inp2, rms_n.weight, 1e-6)


    q1, s1 = per_token_group_quant_fp8(inp3, 128, column_major_scales=True, scale_tma_aligned=True)
    # q1, s1 = sglang_per_token_group_quant_fp8(inp3, 128)
    q2 = torch.empty_like(q1)
    s2 = torch.empty_like(s1)

    fused_rmsnorm_quant(inp1, q2, s2, rms_n.weight, 128, 1e-10, -448.0, 448.0, False)
# q2, s2 = per_token_group_quant_fp8(inp2, 128)


    if "debug" in sys.argv:
        print(torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not().nonzero())
        print(torch.allclose(s2, s2))
        # print(inp3[0])
        # print("")
        print(q1.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()])
        print(q2.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()])
        # # print(q3)
        # print("")
        # print(s1)
        # print(s2)

exit()
for N in [8, 64, 256, 1024, 2048]:
    inp1 = torch.randn((N, H), dtype=torch.bfloat16)
    inp2 = torch.randn((N, H), dtype=torch.bfloat16)
    inp3 = inp1.clone()
    inp4 = inp2.clone()

    rms_n = RMSNorm(7168).to(torch.bfloat16)
    out, res = rms_n.forward_cuda(inp1, inp2)
    # cu_ext.rms_norm(inp1, inp2, rms_n.weight, 1e-6)


    q1, s1 = per_token_group_quant_fp8(out, 128, column_major_scales=True, scale_tma_aligned=True)
    # q1, s1 = sglang_per_token_group_quant_fp8(out, 128)
    # q1, s1, = out
    q2 = torch.empty_like(q1)
    s2 = torch.empty_like(s1)
    print(s2.shape)

    cu_ext.rms_norm_quant_add(inp3, inp4, q2, s2, rms_n.weight, 1e-10)
# q2, s2 = per_token_group_quant_fp8(inp2, 128)


    # if "debug" in sys.argv:
    #     print(torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not().nonzero())
    #     print(torch.allclose(s2, s2))
    #     print(torch.allclose(res, inp4))
    #     # print(inp3[0])
    #     # print("")
    #     print(q1.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()])
    #     print(q2.to(torch.bfloat16)[torch.isclose(q1.to(torch.bfloat16), q2.to(torch.bfloat16)).logical_not()])
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
