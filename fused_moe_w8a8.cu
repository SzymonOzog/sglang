#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const fp8* __restrict__ w,
        const float* __restrict__ w_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N
        )
{
    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[blockIdx.y];
    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const int w_col = blockIdx.x * BN + (lane_id>>2);

    if(blockIdx.y * BM >= num_tokens_post_padded[0])
        return;

    if(exp_idx < 0 || exp_idx >= 257)
        printf("INVALID IDX %d, %d, %d\n",blockIdx.y, exp_idx, num_tokens_post_padded[0]);

    int token_srcs[2];
    token_srcs[0] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2)];
    token_srcs[1] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2) + 8];
    int token_offs[2];
    token_offs[0] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2)] / top_k;
    token_offs[1] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2) + 8] / top_k;

    uint32_t tile_x[4];
    uint32_t tile_w[2];
    float f_acc[4] = {0.f};

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
        const int scale_cols_x = K/block_shape[1];
        const int scale_cols_w = N/block_shape[1];

        float scale_x[2];
        if (token_offs[0] < M)
        {
            scale_x[0] = x_scale[(token_offs[0]/block_shape[0])*scale_cols_x + block];
        }
        if (token_offs[1] < M)
        {
            scale_x[1] = x_scale[(token_offs[1]/block_shape[0])*scale_cols_x + block];
        }

        float scale_w = w_scale[block*scale_cols_w + w_col/block_shape[1]];

        int b_off = block * block_shape[0];
        float acc[4] = {0.f};
        for(int k = 0; k < block_shape[0]; k += BK)
        {
            if (token_offs[0] < M)
            {
                tile_x[0] = reinterpret_cast<const uint32_t*>(x + token_offs[0]*K + k + b_off)[lane_id%4];
                tile_x[2] = reinterpret_cast<const uint32_t*>(x + token_offs[0]*K + k + b_off + 16)[lane_id%4];
            }
            if (token_offs[1] < M)
            {
                tile_x[1] = reinterpret_cast<const uint32_t*>(x + token_offs[1]*K + k + b_off)[lane_id%4];
                tile_x[3] = reinterpret_cast<const uint32_t*>(x + token_offs[1]*K + k + b_off + 16)[lane_id%4];
            }

            fp8 tmp[4];
            for (int i = 0; i<4; i++)
            {
                const int w_row = (lane_id%4) + i + k + b_off;
                tmp[i] = exp_w[w_row*N + w_col];
            }
            tile_w[0] = *reinterpret_cast<uint32_t*>(&tmp);
            for (int i = 0; i<4; i++)
            {
                const int w_row = (lane_id%4) + i + k + b_off + 16;
                tmp[i] = exp_w[w_row*N + w_col];
            }
            tile_w[1] = *reinterpret_cast<uint32_t*>(&tmp);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(tile_x[0]), "r"(tile_x[1]), "r"(tile_x[2]), "r"(tile_x[3]), "r"(tile_w[0]), "r"(tile_w[1]));
            float x_dq[8];
            for (int i = 0; i < 4; i++)
            {
                x_dq[i] = float(reinterpret_cast<fp8*>(&tile_x[0])[i]) * scale_x[0];
                x_dq[4 + i] = float(reinterpret_cast<fp8*>(&tile_x[2])[i]) * scale_x[0];
            }
            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
                printf("mma %d, %d with %f,%f,%f,%f ||| %f, %f, %f, %f acc %f, %f, %f, %f, scale x %f,%f scale_w %f\n", k,
                        token_offs[0],
                        x_dq[0],
                        x_dq[1],
                        x_dq[2],
                        x_dq[3],
                        x_dq[4],
                        x_dq[5],
                        x_dq[6],
                        x_dq[7],
                        acc[0],
                        acc[1],
                        acc[2],
                        acc[3],
                        scale_x[0],
                        scale_x[1],
                        scale_w
                        );
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            //     printf("mma %d with %d/%d, %d,%d,%d,%d, acc %f, %f, %f, %f, scale x %f,%f scale_w %f\n", block,
            //             token_offs[0], token_offs[1],
            //             tile_x[0],
            //             tile_x[2],
            //             tile_w[0],
            //             tile_w[1],
            //             acc[0],
            //             acc[1],
            //             acc[2],
            //             acc[3],
            //             scale_x[0],
            //             scale_x[1],
            //             scale_w
            //             );
        }
        f_acc[0] += scale_x[0] * scale_w * acc[0];
        f_acc[1] += scale_x[0] * scale_w * acc[1];
        f_acc[2] += scale_x[1] * scale_w * acc[2];
        f_acc[3] += scale_x[1] * scale_w * acc[3];
    }
    if (token_offs[0] < M)
    {
        out[token_srcs[0]*N + blockIdx.x * BN + (lane_id%4)*2] = f_acc[0];
        out[token_srcs[0]*N + blockIdx.x * BN + (lane_id%4)*2 + 1] = f_acc[1];
    }
    if (token_offs[1] < M)
    {
        out[token_srcs[1]*N + (lane_id%4)*2] = f_acc[2];
        out[token_srcs[1]*N + (lane_id%4)*2 + 1] = f_acc[3];
    }
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        printf("finished with %d/%d, %f,%f,%f,%f\n", token_offs[0], token_offs[1],
                f_acc[0],
                f_acc[1],
                f_acc[2],
                f_acc[3]);
}

void fused_moe_w8a8(
        const fp8* x,
        const float* x_scale,
        const fp8* w, const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num
        )
{
    constexpr int BM = 16;
    constexpr int BK = 32;
    constexpr int BN = 8;
    dim3 dimBlock(32,1,1);
    dim3 dimGrid(std::ceil((float)N/BN), std::ceil((float)sorted_num/BM), 1);
    printf("launching %d, %d \n", dimGrid.x, dimGrid.y);
    fused_moe_w8a8_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
            x,
            x_scale,
            w,
            w_scale,
            out,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            M,
            K,
            N
            );
}
