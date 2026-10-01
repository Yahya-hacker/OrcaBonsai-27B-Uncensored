# Building the CUDA kernels

Written for the project's target — **RTX 5060 Laptop GPU, 8 GB, Blackwell `sm_120`** —
but the only part specific to that card is the architecture number. Substitute your own
and everything else holds.

> **Status.** The kernels in `bonsai/kernels/` have never been compiled: they were
> authored on a machine with no GPU. Their *arithmetic* is verified on CPU
> (`tests/test_kernel_semantics.py` reimplements the device functions in numpy and
> checks them against the reference codec), but expect to fix compile errors on first
> build. Treat this as a guide to the toolchain, not a promise that it builds clean.

---

## 1. The one thing that goes wrong

Blackwell consumer GPUs are compute capability **12.0** (`sm_120`). Almost every
failure traces back to a toolchain predating it.

| Requirement | Minimum | Recommended |
|---|---|---|
| NVIDIA driver | **R570** | latest |
| CUDA Toolkit | **12.8** | **12.9** |
| PyTorch | 2.7 + cu128 | latest + cu128/cu129 |
| GCC (Linux) | 11 | 12–13 |
| MSVC (Windows) | VS 2022 | — |

The failure mode is nasty because it is **deferred**. An older toolkit compiles happily
for `sm_90`, installs, imports — and then dies at the first kernel launch with:

```
CUDA error: no kernel image is available for execution on the device
```

That message never mentions the toolkit. `bonsai/kernels/check_toolchain()` catches it
up front and says so in plain language.

**The Ubuntu trap.** `apt install nvidia-cuda-toolkit` on Ubuntu 22.04/24.04 installs
**CUDA 12.0**, which predates Blackwell. Its `nvcc` rejects `compute_120` outright:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

Install from NVIDIA's repo instead (§3), not from the distro default.

---

## 2. Check what you have

```bash
nvidia-smi                                    # driver >= 570
nvcc --version                                # release >= 12.8
nvcc --list-gpu-arch | grep 120               # must print compute_120
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```

Expect `(12, 0)` from the last one. Then let the project check it for you:

```bash
python -c "from bonsai.kernels import check_toolchain; print(check_toolchain())"
```

---

## 3. Install the toolchain

### Linux (Ubuntu 22.04 / 24.04) — the supported path

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-9        # NOT nvidia-cuda-toolkit

echo 'export CUDA_HOME=/usr/local/cuda-12.9' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH'      >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Keep the driver from your distro or NVIDIA's `.run`; the toolkit does not have to
install a driver, and on a laptop you usually do not want it to.

### Windows

Native Windows builds for `sm_120` are painful — MSVC version coupling, DLL hell, and
PyTorch's Windows wheels have historically lagged. **Use WSL2.** Install the Windows
NVIDIA driver (which exposes the GPU to WSL), then follow the Linux steps *inside* WSL.
Do **not** install a driver inside WSL.

### PyTorch

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

You do **not** need to build PyTorch from source. Stable wheels have shipped `sm_120`
cubins since the cu128 line; building from source is a last resort for an unusual stack.

---

## 4. Build

### Just in time (what you want while developing)

Nothing to do. The first call compiles into `~/.cache/torch_extensions` and later ones
hit the cache:

```python
from bonsai.kernels import load
k = load(verbose=True)          # ~1-2 min the first time
```

### Ahead of time

```bash
pip install -e .
```

`setup.py` auto-detects your card's capability. Pin it when cross-compiling or when no
GPU is visible at build time:

```bash
TORCH_CUDA_ARCH_LIST="12.0" pip install -e .
```

### Flags, and why

```
-gencode arch=compute_120,code=sm_120      # native cubin for Blackwell
-gencode arch=compute_120,code=compute_120 # PTX too, so future cards JIT forward
-O3 --use_fast_math -std=c++17
--expt-relaxed-constexpr --expt-extended-lambda
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__   # and friends
-lineinfo                                   # source lines in ncu/compute-sanitizer
```

The `-U__CUDA_NO_*` undefines matter. PyTorch defines those macros to stop you using
`__half`/`__nv_bfloat16` arithmetic operators; our kernels use them, so they must be
undefined or you get a wall of "no operator matches" errors.

`TORCH_CUDA_ARCH_LIST="12.0"` is strongly preferred over leaving it unset. Unset, nvcc
compiles for a long default list — minutes of build time, and on a 45 W laptop, real
heat.

A laptop also wants `MAX_JOBS` bounded, or parallel nvcc will thermally throttle the
machine it is building for:

```bash
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" pip install -e .
```

---

## 5. Verify correctness before trusting speed

```bash
pytest tests/ -q                             # 81 CPU tests, must pass first
python -c "from bonsai.kernels import load; load(verbose=True)"
pytest tests/ -q -k gpu                      # GPU parity (added in M1)
```

The parity tests check every kernel against the **Tier-1 oracle** — a per-matrix fp32
reconstruction done in torch:

```python
from bonsai.format import BonsaiReader
r = BonsaiReader("bonsai-27b.bonsai")
W = r.dequantize("model.layers.0.mlp.down_proj")     # exact, 340 MiB
assert torch.allclose(kernels.ternary_gemv(codes, scales, x), x @ W.T, rtol=2e-3)
```

This model produces **fluent nonsense** rather than crashing when a kernel is subtly
wrong, so never skip straight to benchmarking. A plausible-looking paragraph is not
evidence of correctness.

### Tools worth knowing

```bash
compute-sanitizer --tool memcheck python your_script.py    # OOB / races
ncu --set full -o prof python your_script.py               # occupancy, memory throughput
nsys profile -o trace python your_script.py                # launch gaps, graph capture
```

For the decode GEMV the number that matters is **achieved DRAM throughput** as a
fraction of 384–448 GB/s, not FLOPs. Decode is bandwidth-bound: if you are below ~50%
of peak bandwidth, nothing about arithmetic will save you.

---

## 6. Failure lookup

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available` | built for the wrong arch | `TORCH_CUDA_ARCH_LIST="12.0"`, rebuild, clear `~/.cache/torch_extensions` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | CUDA < 12.8 | install `cuda-toolkit-12-9`; remove `nvidia-cuda-toolkit` |
| `CUDA_HOME not set` | toolkit not on PATH | `export CUDA_HOME=/usr/local/cuda-12.9` |
| `no operator "*" matches __half` | PyTorch's `__CUDA_NO_HALF_*` | keep the `-U__CUDA_NO_*` flags |
| `unsupported GNU version` | GCC too new for nvcc | `export CC=gcc-12 CXX=g++-12` |
| ninja not found | missing build tool | `pip install ninja` (also much faster) |
| builds, then hangs | compiling the full arch list | pin `TORCH_CUDA_ARCH_LIST` |
| `out of memory` during build | parallel nvcc | `MAX_JOBS=2` |
| silently wrong output | wrong kernel, not wrong build | run the Tier-1 oracle parity tests |

### Clearing the JIT cache

```bash
rm -rf ~/.cache/torch_extensions/*/bonsai_kernels
```

Stale cached builds survive source edits more often than they should. When a change
"does nothing", clear this first.

---

## 7. Also building llama.cpp?

Worth doing — the PrismML fork is this project's end-to-end oracle (Tier 3) and the
performance baseline the native runtime has to beat.

```bash
git clone https://github.com/PrismML-Eng/llama.cpp -b prism && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j4
```

`120` is the same `sm_120`. Stock llama.cpp **cannot** read these files — the ternary
types 142/143 are private ids beyond upstream's enum — so the fork is mandatory.
