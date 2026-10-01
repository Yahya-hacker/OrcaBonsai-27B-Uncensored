// PyTorch bindings for the native Bonsai kernels.
//
// Deliberately thin: tensor validation and dtype dispatch only. The kernels themselves
// know nothing about torch, so they stay testable from a plain CUDA harness and could
// be lifted into a C++ engine later without untangling them from Python.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace bonsai {
void ternary_gemv_launch(const uint8_t*, const void*, const void*, void*,
                         int, int, bool, cudaStream_t);
void residual_ablate_launch(const void*, const void*, const float*, const float*,
                            void*, int, int, bool, cudaStream_t);
void embed_ablate_launch(const void*, const float*, const float*, void*,
                         int, int, bool, cudaStream_t);
}  // namespace bonsai

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

static bool is_bf16(const torch::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == torch::kHalf || t.scalar_type() == torch::kBFloat16,
                "activations must be float16 or bfloat16, got ", t.scalar_type());
    return t.scalar_type() == torch::kBFloat16;
}

// y = W @ x, W held as .bonsai superblocks.
//   codes  [rows, (in/1024)*208] uint8      scales [rows, in/128] float16
//   x      [in]  half|bfloat16
torch::Tensor ternary_gemv(torch::Tensor codes, torch::Tensor scales, torch::Tensor x,
                           int64_t rows, int64_t in_features) {
    CHECK_CUDA(codes); CHECK_CUDA(scales); CHECK_CUDA(x);
    CHECK_CONTIG(codes); CHECK_CONTIG(scales); CHECK_CONTIG(x);
    TORCH_CHECK(codes.scalar_type() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be float16");
    TORCH_CHECK(in_features % 1024 == 0, "in_features must be a multiple of 1024");
    TORCH_CHECK(x.numel() == in_features, "x has ", x.numel(), " elements, expected ",
                in_features);
    TORCH_CHECK(codes.numel() == rows * (in_features / 1024) * 208,
                "codes size does not match [rows, in]; wrong layout?");
    TORCH_CHECK(scales.numel() == rows * (in_features / 128), "scales size mismatch");

    auto y = torch::empty({rows}, x.options());
    bonsai::ternary_gemv_launch(
        codes.data_ptr<uint8_t>(), scales.data_ptr(), x.data_ptr(), y.data_ptr(),
        (int)rows, (int)in_features, is_bf16(x),
        c10::cuda::getCurrentCUDAStream());
    return y;
}

// h = x + y - alpha * dot(y, r) * r   -- the fused ablation, one block per token.
torch::Tensor residual_ablate(torch::Tensor x, torch::Tensor y, torch::Tensor r,
                              torch::Tensor alpha) {
    CHECK_CUDA(x); CHECK_CUDA(y); CHECK_CUDA(r); CHECK_CUDA(alpha);
    CHECK_CONTIG(x); CHECK_CONTIG(y); CHECK_CONTIG(r);
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must share a dtype");
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "direction must be float32 (R4)");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 && alpha.numel() == 1,
                "alpha must be a 1-element float32 tensor -- it is passed by pointer so "
                "a captured CUDA graph stays tunable (R9)");
    const int64_t hidden = x.size(-1);
    TORCH_CHECK(r.numel() == hidden, "direction length ", r.numel(),
                " does not match hidden ", hidden);
    TORCH_CHECK(hidden <= 8192, "hidden ", hidden, " exceeds the per-thread register "
                "budget (kMaxPerThread=8 at 1024 threads)");

    auto out = torch::empty_like(x);
    bonsai::residual_ablate_launch(
        x.data_ptr(), y.data_ptr(), r.data_ptr<float>(), alpha.data_ptr<float>(),
        out.data_ptr(), (int)(x.numel() / hidden), (int)hidden, is_bf16(x),
        c10::cuda::getCurrentCUDAStream());
    return out;
}

// h = e - alpha * dot(e, r) * r   -- the embedding, the one site with no residual in.
torch::Tensor embed_ablate(torch::Tensor e, torch::Tensor r, torch::Tensor alpha) {
    CHECK_CUDA(e); CHECK_CUDA(r); CHECK_CUDA(alpha);
    CHECK_CONTIG(e); CHECK_CONTIG(r);
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "direction must be float32 (R4)");
    const int64_t hidden = e.size(-1);
    TORCH_CHECK(r.numel() == hidden, "direction length mismatch");

    auto out = torch::empty_like(e);
    bonsai::embed_ablate_launch(
        e.data_ptr(), r.data_ptr<float>(), alpha.data_ptr<float>(), out.data_ptr(),
        (int)(e.numel() / hidden), (int)hidden, is_bf16(e),
        c10::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native CUDA kernels for Ternary Bonsai 2 27B";
    m.def("ternary_gemv", &ternary_gemv, "ternary GEMV (decode)",
          py::arg("codes"), py::arg("scales"), py::arg("x"),
          py::arg("rows"), py::arg("in_features"));
    m.def("residual_ablate", &residual_ablate, "fused residual add + refusal ablation",
          py::arg("x"), py::arg("y"), py::arg("r"), py::arg("alpha"));
    m.def("embed_ablate", &embed_ablate, "refusal ablation on embedding output",
          py::arg("e"), py::arg("r"), py::arg("alpha"));
}
