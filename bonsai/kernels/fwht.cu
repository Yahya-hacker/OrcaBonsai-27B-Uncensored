// K3 -- fused RMSNorm + sign + fast Walsh-Hadamard transform, block 1024.
//
// STATUS: reference implementation, never compiled. Arithmetic mirrors
// bonsai/reference.py::fwht, which is tested.
//
// The optimisation MLX leaves on the table
// ----------------------------------------
// The pack folds its Hadamard rotation into each matrix's *input* dimension, so every
// packed projection must transform its input before the matmul. MLX's Packed.__call__
// therefore runs the transform inside every projection -- but the transform depends
// only on the input vector, and several projections share one:
//
//     q_proj, k_proj, v_proj                  <- same input  (3 transforms -> 1)
//     gate_proj, up_proj                      <- same input  (2 -> 1)
//     in_proj_qkv, in_proj_z, in_proj_a/b     <- same input  (4 -> 1)
//
// Hoisting it to once per distinct input takes a layer from ~10 transforms to 5. Since
// the transform is also always preceded by an RMSNorm, the two fuse: normalise, apply
// signs, transform, all in one pass with the vector resident in shared memory.
//
// Shape
// -----
// hidden 5120 = 5 blocks of 1024. One CUDA block handles one 1024-element block for one
// token: 1024 floats = 4 KB of shared memory, 10 butterfly stages, 512 butterflies per
// stage across 512 threads.
//
// Normalisation is 1/sqrt(1024) = 1/32, which makes the transform orthogonal (so it is
// its own inverse) and is also what keeps fp16 activations in range at 27B scale (R8).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace bonsai {

constexpr int kHadBlock = 1024;
constexpr int kHadThreads = kHadBlock / 2;     // one thread per butterfly
constexpr int kWarpF = 32;

// grid: (blocks_per_row, tokens) -- blockIdx.x selects the 1024-slice, .y the token.
//
//   x       [tokens, hidden]
//   signs   [hidden]    +-1 per element, or null
//   weight  [hidden]    RMSNorm gain, or null to skip the norm
//   inverse false: signs then transform (forward, used by every projection)
//          true : transform then signs (used by the embedding, R5)
template <typename T>
__global__ void rmsnorm_sign_fwht(
        const T* __restrict__ x, const float* __restrict__ signs,
        const float* __restrict__ weight, T* __restrict__ out,
        int hidden, float eps, bool inverse) {
    __shared__ float buf[kHadBlock];
    __shared__ float red[kWarpF];

    const long token = blockIdx.y;
    const int  slice = blockIdx.x;
    const long base  = token * hidden + (long)slice * kHadBlock;
    const int  tid   = threadIdx.x;

    // ---- load the 1024-slice (two elements per thread)
    buf[tid] = static_cast<float>(x[base + tid]);
    buf[tid + kHadThreads] = static_cast<float>(x[base + tid + kHadThreads]);
    __syncthreads();

    // ---- RMSNorm over the WHOLE hidden vector, not just this slice.
    // Caller must pass hidden == kHadBlock to fuse the norm; otherwise pass
    // weight == nullptr and normalise in a prior pass. Getting this wrong would
    // normalise each 1024-slice independently, which is a different operator.
    if (weight != nullptr) {
        float acc = buf[tid] * buf[tid]
                  + buf[tid + kHadThreads] * buf[tid + kHadThreads];
        #pragma unroll
        for (int off = kWarpF / 2; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if ((tid & (kWarpF - 1)) == 0) red[tid / kWarpF] = acc;
        __syncthreads();
        if (tid < kWarpF) {
            float v = (tid < blockDim.x / kWarpF) ? red[tid] : 0.0f;
            #pragma unroll
            for (int off = kWarpF / 2; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, off);
            if (tid == 0) red[0] = rsqrtf(v / kHadBlock + eps);
        }
        __syncthreads();
        const float inv = red[0];
        buf[tid] *= inv * weight[slice * kHadBlock + tid];
        buf[tid + kHadThreads] *=
            inv * weight[slice * kHadBlock + tid + kHadThreads];
        __syncthreads();
    }

    // ---- signs before the transform (forward only)
    if (signs != nullptr && !inverse) {
        buf[tid] *= signs[slice * kHadBlock + tid];
        buf[tid + kHadThreads] *= signs[slice * kHadBlock + tid + kHadThreads];
        __syncthreads();
    }

    // ---- 10 butterfly stages. Thread tid owns butterfly tid at every stage; the
    // index arithmetic splits tid into (block above the stride, offset within it).
    #pragma unroll
    for (int h = 1; h < kHadBlock; h <<= 1) {
        const int lo = ((tid / h) * 2 * h) + (tid % h);
        const int hi = lo + h;
        const float a = buf[lo], b = buf[hi];
        __syncthreads();
        buf[lo] = a + b;
        buf[hi] = a - b;
        __syncthreads();
    }

    const float norm = rsqrtf((float)kHadBlock);           // 1/32 at block 1024
    float v0 = buf[tid] * norm;
    float v1 = buf[tid + kHadThreads] * norm;

    // ---- signs after the transform (inverse only, the embedding path)
    if (signs != nullptr && inverse) {
        v0 *= signs[slice * kHadBlock + tid];
        v1 *= signs[slice * kHadBlock + tid + kHadThreads];
    }
    out[base + tid] = static_cast<T>(v0);
    out[base + tid + kHadThreads] = static_cast<T>(v1);
}

template __global__ void rmsnorm_sign_fwht<__half>(
    const __half*, const float*, const float*, __half*, int, float, bool);
template __global__ void rmsnorm_sign_fwht<__nv_bfloat16>(
    const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, int, float, bool);

void fwht_launch(const void* x, const float* signs, const float* weight, void* out,
                 int tokens, int hidden, float eps, bool inverse, bool bf16,
                 cudaStream_t stream) {
    dim3 grid(hidden / kHadBlock, tokens);
    if (bf16) {
        rmsnorm_sign_fwht<__nv_bfloat16><<<grid, kHadThreads, 0, stream>>>(
            (const __nv_bfloat16*)x, signs, weight, (__nv_bfloat16*)out,
            hidden, eps, inverse);
    } else {
        rmsnorm_sign_fwht<__half><<<grid, kHadThreads, 0, stream>>>(
            (const __half*)x, signs, weight, (__half*)out, hidden, eps, inverse);
    }
}

}  // namespace bonsai
