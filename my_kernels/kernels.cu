#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#define RMS_BLOCK_SIZE 1024

// like std::array, but aligned
// goal: generate ld.128 and st.128 instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};


template <typename scalar_t, bool FUSED_ADD>
__global__ void rms_norm_kernel(scalar_t* input, scalar_t* weight, scalar_t* output, const unsigned int d, const unsigned int rows, const float eps)
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
            P a = reinterpret_cast<P*>(output)[row * d + idx];
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
    for(int64_t idx = tx; idx<d; idx+=blockDim.x)
    {
        P x = reinterpret_cast<P*>(input)[row * d + idx];
        P w = reinterpret_cast<P*>(weight)[idx];
        P out;
        for (int64_t i = 0; i<P::size; i++)
        {
            out.data[i] = (float)x.data[i] * acc;
            out.data[i] *= (float)w.data[i];
        }
        reinterpret_cast<P*>(output)[row * d + idx] = out;
    }

}

void rms_norm_launcher(torch::Tensor& input, torch::Tensor& output, torch::Tensor& weight, float eps)
{
    const unsigned int d = input.size(-1);
    const unsigned int rows = input.size(-2);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rms_norm_add", ([&] {

                const unsigned int packed_d = std::ceil((float)d * sizeof(scalar_t) / 16);

                dim3 block_size = dim3(RMS_BLOCK_SIZE, 1, 1);
                dim3 grid_size = dim3(rows, 1, 1);

                const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
                const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                rms_norm_kernel<scalar_t, false><<<grid_size, block_size, 0, stream>>>
                (input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), packed_d, rows, eps);
                }));
}

