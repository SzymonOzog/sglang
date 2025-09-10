#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>


void fused_moe_w8a8(
        const __nv_fp8_e4m3* x,
        const float* x_scale,
        const __nv_fp8_e4m3* w,
        const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num
        );

torch::Tensor fused_moe_launcher(
        torch::Tensor& x,
        torch::Tensor& x_scale,
        torch::Tensor& w,
        torch::Tensor& w_scale,
        torch::Tensor& sorted_token_ids,
        torch::Tensor& expert_ids,
        torch::Tensor& num_tokens_post_padded,
        int top_k
        )
{
    auto options = torch::TensorOptions().dtype(at::ScalarType::BFloat16).device(w.device());
    torch::Tensor out = torch::empty({x.size(0) * top_k, w.size(1)}, options);
    fused_moe_w8a8 (
            static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0)
            );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_moe_w8a8", &fused_moe_launcher);
}
