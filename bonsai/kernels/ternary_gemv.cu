// K1 -- ternary GEMV for decode. The bandwidth-bound kernel that sets token rate.
//
// STATUS: reference implementation. Correct by construction and written to be checked
// against the Tier-1 oracle (bonsai.format.BonsaiReader.dequantize). It has never been
// compiled or profiled, because the machine this was authored on has no GPU. Treat the
// performance notes as hypotheses to measure in M1, not as claims.
//
// Layout it consumes (see bonsai/format.py)
// -----------------------------------------
//   codes   [out, in/1024, 208]  uint8, 8 groups of 26 bytes, 16-byte aligned
//   scales  [out, in/128]        fp16, separate plane so both loads are coalesced
//   no bias plane: bias == -scale holds exactly, so w = (code - 1) * scale
//
// Trit decode
// -----------
// A byte b holds 5 trits as the base-3 expansion of the fraction b/256:
//
//     trit_t = ((b * 3^t) & 255) * 3 >> 8
//
// Three stages per 128-group -- bytes [0,16) hold 5 trits each, [16,24) hold 5, [24,26)
// hold 4 -- giving 16*5 + 8*5 + 2*4 = 128. The element each trit belongs to is fixed by
// the stage, computed in trit_index() below.
//
// Why a scatter is fine
// ---------------------
// We are accumulating a dot product, so weights may be consumed in any order provided
// each is paired with the right activation. That frees us from reordering trits at
// convert time (which would risk a silent re-encoding bug) -- the kernel just indexes
// x through the stage map. x lives in shared memory, so the scatter is a shared-memory
// read, not a global one.
//
// Known optimisation path for M1, deliberately not taken yet:
//   * read the 208-byte superblock as 13 x uint4 instead of byte-per-lane (26 of 32
//     lanes are active in this version -- 81% load efficiency);
//   * stage x in registers per warp rather than re-reading shared memory per trit;
//   * __ldg / cp.async pipelining of the next superblock.
// Each of those is a measurement away from being justified, and none changes the
// numerics, so they belong after the oracle says this version is right.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace bonsai {

constexpr int kGroup       = 128;
constexpr int kSuperGroups = 8;
constexpr int kGroupBytes  = 26;
constexpr int kSuperBytes  = kGroupBytes * kSuperGroups;   // 208
constexpr int kSuperWeights = kGroup * kSuperGroups;       // 1024
constexpr int kWarp        = 32;

// (byte, trit) -> element index inside its 128-group.
__device__ __forceinline__ int trit_index(int byte, int trit) {
    if (byte < 16) return trit * 16 + byte;                 // stage 0: 16 bytes x 5
    if (byte < 24) return 80 + trit * 8 + (byte - 16);      // stage 1:  8 bytes x 5
    return 120 + trit * 2 + (byte - 24);                    // stage 2:  2 bytes x 4
}
__device__ __forceinline__ int trits_in_byte(int byte) { return byte < 24 ? 5 : 4; }

// y[row] = sum_k w[row,k] * x[k], one warp per output row.
//
//   codes   [rows, in/1024 * 208]
//   scales  [rows, in/128]
//   x       [in]   staged into shared memory by the block
//
// grid.x covers rows in steps of (blockDim.x / 32).
template <typename T>
__global__ void ternary_gemv(
        const uint8_t* __restrict__ codes, const __half* __restrict__ scales,
        const T* __restrict__ x, T* __restrict__ y, int rows, int in_features) {
    extern __shared__ float sx[];                 // in_features floats

    for (int i = threadIdx.x; i < in_features; i += blockDim.x)
        sx[i] = static_cast<float>(x[i]);
    __syncthreads();

    const int lane     = threadIdx.x & (kWarp - 1);
    const int warp     = threadIdx.x / kWarp;
    const int warps    = blockDim.x / kWarp;
    const int supers   = in_features / kSuperWeights;
    const int n_groups = in_features / kGroup;

    for (int row = blockIdx.x * warps + warp; row < rows; row += gridDim.x * warps) {
        const uint8_t* crow = codes  + (size_t)row * supers * kSuperBytes;
        const __half*  srow = scales + (size_t)row * n_groups;
        float acc = 0.0f;

        for (int sb = 0; sb < supers; ++sb) {
            const uint8_t* cs = crow + sb * kSuperBytes;
            #pragma unroll
            for (int g = 0; g < kSuperGroups; ++g) {
                const int gidx = sb * kSuperGroups + g;
                const float scale = __half2float(srow[gidx]);
                const int xbase = gidx * kGroup;
                // 26 data bytes per group; lanes 26..31 idle (see M1 notes above).
                float part = 0.0f;
                if (lane < kGroupBytes) {
                    const unsigned b = cs[g * kGroupBytes + lane];
                    const int nt = trits_in_byte(lane);
                    unsigned p = b;
                    #pragma unroll
                    for (int t = 0; t < 5; ++t) {
                        if (t < nt) {
                            // ((b * 3^t) & 255) * 3 >> 8, carried incrementally
                            const int code = (int)(((p & 255u) * 3u) >> 8);
                            part = fmaf((float)(code - 1),
                                        sx[xbase + trit_index(lane, t)], part);
                            p = (p * 3u) & 255u;
                        }
                    }
                }
                #pragma unroll
                for (int off = kWarp / 2; off > 0; off >>= 1)
                    part += __shfl_xor_sync(0xffffffffu, part, off);
                if (lane == 0) acc = fmaf(part, scale, acc);
            }
        }
        if (lane == 0) y[row] = static_cast<T>(acc);
    }
}

template __global__ void ternary_gemv<__half>(
    const uint8_t*, const __half*, const __half*, __half*, int, int);
template __global__ void ternary_gemv<__nv_bfloat16>(
    const uint8_t*, const __half*, const __nv_bfloat16*, __nv_bfloat16*, int, int);

}  // namespace bonsai
