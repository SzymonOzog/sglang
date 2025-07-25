#include <cstdint>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void rms_norm_launcher(torch::Tensor& input, torch::Tensor& output, torch::Tensor& weight, float eps);

void rms_norm(torch::Tensor& input, torch::Tensor& output, torch::Tensor& weight, double eps)
{
    rms_norm_launcher(input, output, weight, eps);
}

void rms_norm_quant_launcher(torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s,
        torch::Tensor& weight, double eps);

void rms_norm_quant(torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s,
        torch::Tensor& weight, double eps)
{
    rms_norm_quant_launcher(input, output_q, output_s, weight, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rms_norm", &rms_norm, "rms_norm");
  // m.def("quantize", &quantize, "quantize");
  m.def("rms_norm_quant", &rms_norm_quant, "rms_norm_quant");
}
