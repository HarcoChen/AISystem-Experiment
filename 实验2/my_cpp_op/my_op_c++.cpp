#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")//强制CUDA
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor BentForward(torch::Tensor input)
{
    CHECK_INPUT(input);
    torch::Tensor output=input+(torch::sqrt(input*input+1)-1)/2;
    return output;
}
torch::Tensor BentBackward(torch::Tensor grad_out,torch::Tensor input)
{
    torch::Tensor grad_x=1+input/(2*torch::sqrt(input*input+1));
    return grad_out*grad_x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &BentForward, "BentIdentity forward (C++)");
    m.def("backward", &BentBackward, "BentIdentity backward (C++)");
}