
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <flashinfer/vec_dtypes.cuh>

using FP8_TYPE = c10::Float8_e4m3fn;
#define RMS_BLOCK_SIZE 1024

// like std::array, but aligned
// goal: generate ld.128 and st.128 instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
//   unsigned mask = 0xffff;
//
//   val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
//   val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
//   val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
//   val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
//   return val;
// }

constexpr float max_8_bit = 448.0;
constexpr float min_8_bit = -448.0;

template <typename scalar_t, bool FUSED_ADD>
__global__ void rms_norm_quant_kernel(scalar_t* __restrict__  input, scalar_t* __restrict__  weight,
        FP8_TYPE* __restrict__  output_q, float* __restrict__ output_s,
        const unsigned int d, const unsigned int rows, const float eps)
{
    int64_t row = blockIdx.x;
    int64_t tx = threadIdx.x;
    int64_t warp_id = tx/32;
    using P = array_t<scalar_t, 16 / sizeof(scalar_t)>;
    float acc = 0.f;
    __shared__ float reduction[RMS_BLOCK_SIZE/32];
    for(int64_t idx = tx; idx<d; idx+=blockDim.x)
    {
        P x = reinterpret_cast<P*>(input)[row * d + idx];

        if constexpr(FUSED_ADD)
        {
            P a = reinterpret_cast<P*>(output_q)[row * d + idx];
            for (int64_t i = 0; i<P::size; i++)
            {
                x.data[i] += a.data[i];
            }
            reinterpret_cast<P*>(input)[row * d + idx] = x;
        }

        for (int64_t i = 0; i<P::size; i++)
        {
            acc += (float)x.data[i] * (float)x.data[i];
        }

    }
    for (int mask = 16; mask>0; mask/=2)
    {
      acc += __shfl_xor_sync(0xffffffff, acc, mask, 32);
    }

    if(threadIdx.x%32 == 0)
    {
        reduction[warp_id] = acc;
    }

    __syncthreads();

    if (warp_id == 0)
    {
        acc = tx < RMS_BLOCK_SIZE/32 ? reduction[tx] : 0.f;
        for (int mask = 16; mask>0; mask/=2)
        {
          acc += __shfl_xor_sync(0xffffffff, acc, mask, 32);
        }
    }
    if(tx == 0)
    {
        float var = acc/(d*P::size);
        reduction[0] = rsqrtf(var + eps);
    }

    __syncthreads();
    acc = reduction[0];
    // const uint32_t elems_per_thread = P::size;
    using O = array_t<FP8_TYPE, 8 / sizeof(FP8_TYPE)>;
    for(int64_t idx = tx; idx<d; idx+=blockDim.x)
    {
        float local_absmax = eps;
        P x = reinterpret_cast<P*>(input)[row * d + idx];
        P w = reinterpret_cast<P*>(weight)[idx];
        P interm;
        for (int64_t i = 0; i<P::size; i++)
        {
            interm.data[i] = (float)x.data[i] * acc;
            interm.data[i] *= (float)w.data[i];
            local_absmax = fmaxf(local_absmax, fabsf(x.data[i]));
        }
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 8));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 4));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 2));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 1));

        float y_s = (local_absmax/max_8_bit);
        if (threadIdx.x%16 == 0)
        {
             // = y_s;
            __stcg(&output_s[row*(d*P::size/128) + (idx * P::size) / 128], y_s);
        }

        O out;
        for (int64_t i = 0; i<P::size; i++)
        {
            float q = (float)interm.data[i]/y_s;
            float out_val = fminf(fmaxf(q, min_8_bit), max_8_bit);
            out.data[i] = FP8_TYPE(out_val);

        }
        __stcg(&reinterpret_cast<int2*>(output_q)[row * d + idx],
                *reinterpret_cast<int2*>(&out));
    }

}

void rms_norm_quant_launcher(torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s,
        torch::Tensor& weight, double eps)
{
    const unsigned int d = input.size(-1);
    const unsigned int rows = input.size(-2);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rms_norm_quant_add", ([&] {

                const unsigned int packed_d = std::ceil((float)d * sizeof(scalar_t) / 16);

                dim3 block_size = dim3(RMS_BLOCK_SIZE, 1, 1);
                dim3 grid_size = dim3(rows, 1, 1);

                const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
                const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                rms_norm_quant_kernel<scalar_t, false><<<grid_size, block_size, 0, stream>>>
                (input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                 static_cast<FP8_TYPE*>(output_q.data_ptr()),
                 output_s.data_ptr<float>(),
                 packed_d, rows, eps);
                }));
}

