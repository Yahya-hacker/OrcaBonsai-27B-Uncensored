"""Sampling and the generation loop.

Prefill processes the prompt in one pass; decode then runs one token at a time,
carrying a KV cache for the 16 attention layers and a recurrent state for the 48
gated-delta layers. The GDN state is context-independent, so only the KV cache grows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    seed: int | None = None


def sample(logits: torch.Tensor, cfg: SamplingConfig,
           generated: list[int] | None = None,
           gen: torch.Generator | None = None) -> int:
    """Greedy when temperature is 0, else nucleus/top-k sampling."""
    logits = logits.float()

    if cfg.repetition_penalty != 1.0 and generated:
        idx = torch.tensor(sorted(set(generated)), device=logits.device)
        picked = logits[idx]
        logits[idx] = torch.where(picked > 0, picked / cfg.repetition_penalty,
                                  picked * cfg.repetition_penalty)

    if cfg.temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / cfg.temperature

    if cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if 0 < cfg.top_p < 1:
        srt, idx = torch.sort(probs, descending=True)
        cum = srt.cumsum(-1)
        keep = cum - srt <= cfg.top_p          # always keeps the top token
        srt = torch.where(keep, srt, torch.zeros_like(srt))
        srt /= srt.sum()
        return int(idx[torch.multinomial(srt, 1, generator=gen)].item())

    return int(torch.multinomial(probs, 1, generator=gen).item())


@torch.inference_mode()
def generate(model, prompt_ids: list[int], cfg: SamplingConfig | None = None,
             eos_ids: set[int] | None = None,
             caches=None, states=None) -> Iterator[int]:
    """Yield token ids one at a time. Caller owns detokenisation."""
    cfg = cfg or SamplingConfig()
    eos_ids = eos_ids or {model.cfg.eos_token_id}
    device = model.rope_cos.device

    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device=device).manual_seed(cfg.seed)

    caches = model.new_caches() if caches is None else caches
    states = model.new_states() if states is None else states

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = model(ids, caches, states, offset=0)[0, -1]
    offset = len(prompt_ids)

    produced: list[int] = []
    for _ in range(cfg.max_tokens):
        tok = sample(logits, cfg, produced, gen)
        if tok in eos_ids:
            return
        produced.append(tok)
        yield tok
        nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
        logits = model(nxt, caches, states, offset=offset)[0, -1]
        offset += 1


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, cfg: SamplingConfig | None = None,
                  chat: bool = True) -> str:
    if chat and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True)
    else:
        ids = tokenizer.encode(prompt)
    eos = {model.cfg.eos_token_id}
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(tokenizer.eos_token_id)
    return tokenizer.decode(list(generate(model, list(ids), cfg, eos)))
