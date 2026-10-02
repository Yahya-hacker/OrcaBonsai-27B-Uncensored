#!/usr/bin/env python3
"""OrcaBonsai -- Ternary Bonsai 2 27B on NVIDIA, with runtime refusal ablation.

    # one-off
    python run_cuda.py --model bonsai-27b.bonsai --prompt "Explain ternary quantisation."

    # interactive, prefix reuse across turns
    python run_cuda.py --model bonsai-27b.bonsai --chat

    # sweep the ablation strength without reloading
    python run_cuda.py --model bonsai-27b.bonsai --prompt "..." --alpha 0.8

``--alpha`` is the refusal ablation strength: 0 disables it and is bit-identical to the
base model, 1.0 matches a full weight orthogonalisation, above 1 over-projects. It is a
runtime value, so it can change per request with no reload and no modified weights.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a .bonsai container")
    p.add_argument("--direction", default="directions/refusal_dir_fp32.bin",
                   help="refusal direction; 'none' to disable entirely")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer dir or id (defaults to the model's folder)")
    p.add_argument("--prompt")
    p.add_argument("--chat", action="store_true", help="interactive loop")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--no-kernels", action="store_true",
                   help="force the portable torch path (slower, same output)")
    p.add_argument("--raw", action="store_true", help="skip the chat template")
    return p


def main() -> int:
    a = build_parser().parse_args()
    import torch

    from bonsai.generate import SamplingConfig, generate
    from bonsai.loader import load_model

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Run `python tools/doctor.py` to find out why; "
              "falling back to CPU, which will be extremely slow.", file=sys.stderr)
        a.device = "cpu"

    if a.device.startswith("cuda") and not a.no_kernels:
        from bonsai.kernels import available
        if not available():
            print("note: compiled kernels unavailable, using the portable torch path. "
                  "Output is identical, throughput is not. See docs/BUILDING-CUDA.md.",
                  file=sys.stderr)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype]
    direction = None if a.direction.lower() == "none" else a.direction

    t0 = time.time()
    model = load_model(a.model, direction=direction, alpha=a.alpha,
                       device=a.device, dtype=dtype, use_kernels=not a.no_kernels)
    print(f"loaded in {time.time() - t0:.1f}s | alpha={a.alpha} | "
          f"{model.n_ablation_sites()} ablation sites active", file=sys.stderr)

    tok_dir = a.tokenizer or str(Path(a.model).parent)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir)
    except Exception as e:
        print(f"could not load a tokenizer from {tok_dir}: {e}\n"
              "pass --tokenizer with the HF repo or a local folder.", file=sys.stderr)
        return 2

    sampling = SamplingConfig(
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
        repetition_penalty=a.repetition_penalty, max_tokens=a.max_tokens, seed=a.seed)
    eos = {model.cfg.eos_token_id}
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)

    def run(text: str, history=None):
        if a.raw:
            ids = tok.encode(text)
        else:
            msgs = (history or []) + [{"role": "user", "content": text}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=True)
        t = time.time()
        n, pieces = 0, []
        for tid in generate(model, list(ids), sampling, eos):
            s = tok.decode([tid], skip_special_tokens=True)
            pieces.append(s)
            print(s, end="", flush=True)
            n += 1
        dt = time.time() - t
        print(f"\n\n[{n} tokens, {dt:.1f}s, {n / max(dt, 1e-9):.1f} tok/s]",
              file=sys.stderr)
        return "".join(pieces)

    if a.chat:
        history = []
        print("chat mode -- '/alpha 0.5' to retune, '/reset' to clear, Ctrl-D to quit",
              file=sys.stderr)
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not line:
                continue
            if line.startswith("/alpha"):
                model.set_alpha(float(line.split()[1]))
                print(f"alpha = {model.alpha}", file=sys.stderr); continue
            if line == "/reset":
                history = []; print("history cleared", file=sys.stderr); continue
            reply = run(line, history)
            history += [{"role": "user", "content": line},
                        {"role": "assistant", "content": reply}]
    elif a.prompt:
        run(a.prompt)
    else:
        print("give --prompt or --chat", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
