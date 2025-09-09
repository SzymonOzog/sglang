#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

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
        int N
        )
{
    printf("called fused moe yeah \n");
}
