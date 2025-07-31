#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <flashinfer/vec_dtypes.cuh>

using FP8_TYPE = c10::Float8_e4m3fn;
#define RMS_BLOCK_SIZE 256
#define PACK_SIZE 16

// like std::array, but aligned
// goal: generate ld.128 and st.128 instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <
    typename scalar_t,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void rms_norm_quant_kernel(scalar_t* __restrict__  input,
        void* __restrict__  output_q_v,
        scale_packed_t* __restrict__ output_s,
        scalar_t* __restrict__  weight,
        const int32_t group_size,
        const float rms_eps,
        const float quant_eps,
        const float fp8_min,
        const float fp8_max,
        const unsigned int stride,
        const unsigned int s_stride,
        const unsigned int d,
        const unsigned int rows)
{
    DST_DTYPE* output_q = reinterpret_cast<DST_DTYPE*>(output_q_v);
    int64_t row = blockIdx.x;
    int64_t tx = threadIdx.x;
    int64_t warp_id = tx/32;
    using P = array_t<scalar_t, PACK_SIZE / sizeof(scalar_t)>;
    float acc = 0.f;
    __shared__ float reduction[RMS_BLOCK_SIZE/32];
    for(int64_t idx = tx; idx<d; idx+=blockDim.x)
    {
        P x = reinterpret_cast<P*>(input + row*stride)[idx];

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
        acc += __shfl_xor_sync(0xffffffff, acc, 16, 32);
        acc += __shfl_xor_sync(0xffffffff, acc, 8, 32);
        acc += __shfl_xor_sync(0xffffffff, acc, 4, 32);
        acc += __shfl_xor_sync(0xffffffff, acc, 2, 32);
        acc += __shfl_xor_sync(0xffffffff, acc, 1, 32);
    }
    if(tx == 0)
    {
        float var = acc/(d*P::size);
        reduction[0] = rsqrtf(var + rms_eps);
    }

    __syncthreads();
    acc = reduction[0];
    using O = array_t<DST_DTYPE, P::size>;
    for(int64_t idx = tx; idx<d; idx+=blockDim.x)
    {
        float local_absmax = quant_eps;
        P x = reinterpret_cast<P*>(input + row*stride)[idx];
        P w = reinterpret_cast<P*>(weight)[idx];
        P interm;
        for (int64_t i = 0; i<P::size; i++)
        {
            float val = (float)x.data[i] * acc * (float)w.data[i];
            local_absmax = fmaxf(local_absmax, fabsf(val));
            interm.data[i] = val;
        }
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 8));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 4));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 2));
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 1));

        float y_s = (local_absmax/fp8_max);
        if (threadIdx.x%16 == 0)
        {
            int col = (idx * P::size) / group_size;
            const int off = (col)*s_stride + row;
            __stcg(&output_s[off], y_s);
        }

        O out;
        for (int64_t i = 0; i<P::size; i++)
        {
            float q = (float)interm.data[i]/y_s;
            q = fminf(fmaxf(q, fp8_min), fp8_max);
            out.data[i] = DST_DTYPE(q);

        }
        __stcg(&reinterpret_cast<int2*>(output_q)[row * d + idx],
                *reinterpret_cast<int2*>(&out));
    }

}
void rms_norm_quant_launcher(torch::Tensor& input,
        torch::Tensor& output_q,
        torch::Tensor& output_s,
        torch::Tensor& weight,
        int64_t group_size,
        double rms_eps,
        double quant_eps,
        double fp8_min,
        double fp8_max,
        bool scale_ue8m0)
{
    const unsigned int d = input.size(-1);
    const unsigned int rows = input.size(-2);
    const unsigned int stride = input.stride(-2);
    const unsigned int scale_stride = output_s.stride(1);

    dim3 block_size = dim3(RMS_BLOCK_SIZE, 1, 1);
    dim3 grid_size = dim3(rows, 1, 1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
    dim3 grid(RMS_BLOCK_SIZE);                                                                    \
    dim3 block(rows);                                                                             \
    if (is_column_major) {                                                                        \
      if (scale_ue8m0) {                                                                          \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true><<<grid, block, 0, stream>>>(  \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<uint32_t*>(output_s.data_ptr()),                                          \
            static_cast<T*>(weight.data_ptr()),                                                   \
            (int32_t)group_size,                                                                  \
            (float)rms_eps,                                                                       \
            (float)quant_eps,                                                                     \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            packed_d,                                                                             \
            scale_stride,                                                                         \
            rows);                                                                                \
      } else {                                                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false><<<grid, block, 0, stream>>>( \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<float*>(output_s.data_ptr()),                                             \
            static_cast<T*>(weight.data_ptr()),                                                   \
            (int32_t)group_size,                                                                  \
            (float)rms_eps,                                                                       \
            (float)quant_eps,                                                                     \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            packed_d,                                                                             \
            scale_stride,                                                                         \
            rows);                                                                                \
      }                                                                                           \
    } else {                                                                                      \
      assert(!scale_ue8m0);                                                                       \
      rms_norm_quant_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(                     \
        static_cast<T*>(input.data_ptr()),                                                        \
        output_q.data_ptr(),                                                                      \
        static_cast<float*>(output_s.data_ptr()),                                                 \
        static_cast<T*>(weight.data_ptr()),                                                       \
        (int32_t)group_size,                                                                      \
        (float)rms_eps,                                                                           \
        (float)quant_eps,                                                                         \
        (float)min_8bit,                                                                          \
        (float)max_8bit,                                                                          \
        packed_d,                                                                                 \
        scale_stride,                                                                             \
        rows);                                                                                    \
    }                                                                                             \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(scalar_t, int8_t);
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
      return true;
    }
    return false;
  });
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rms_norm_quant_add", ([&] {

                const unsigned int packed_d = std::ceil((float)d * sizeof(scalar_t) / PACK_SIZE);

                const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
                const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                rms_norm_quant_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>
                (input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                 static_cast<FP8_TYPE*>(output_q.data_ptr()),
                 output_s.data_ptr<float>(),
                 packed_d, rows, eps, stride, s_stride);
                }));
}
