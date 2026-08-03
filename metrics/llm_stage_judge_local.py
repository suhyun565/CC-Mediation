"""Local HuggingFace logprob-based DMIS stage classifier.

Mirrors llm_stage_judge.measure_stage_via_logprob but runs forward pass
locally so we can use non-OpenAI models (Llama, Gemma, Mistral, Phi) as
labelers. OpenRouter does NOT pass through logprobs for these families,
so a local labeler is the only way to get a non-OpenAI logprob signal.

The output schema matches measure_stage_via_logprob:
  {
    "distribution":  list[float] over 6 DMIS stages (sums to 1),
    "argmax_idx":    int 0..5,
    "argmax_stage":  str,
    "max_prob":      float,
    "entropy":       float,
    "raw_logprobs":  dict[str, float],
    "error":         Optional[str],
  }
"""

from __future__ import annotations


# --- release bootstrap: make shared/ and metrics/ importable ---
import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("shared", "metrics"):
    _p = str(_RELEASE_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import math
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_stage_judge import JUDGE_SYSTEM, _build_user, STAGE_ORDER


_LOCAL_CACHE: dict[str, tuple] = {}


def _load_local(model_name: str, device: str = "cuda"):
    if model_name in _LOCAL_CACHE:
        return _LOCAL_CACHE[model_name]
    attn = "eager" if "gemma-2" in model_name.lower() else "sdpa"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map=device,
    )
    model.eval()
    _LOCAL_CACHE[model_name] = (model, tok)
    return model, tok


def _digit_token_ids(tokenizer):
    """For each digit 1..6, find the token id that the tokenizer would emit
    after a system+user chat prompt asking for a single digit response. We
    probe both '1' and ' 1' (leading-space) variants and keep whichever
    encodes to a single token. Falls back to multi-token if necessary."""
    out: dict[str, list[int]] = {}
    for d in "123456":
        for variant in (d, " " + d):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[d] = ids
                break
        else:
            out[d] = tokenizer.encode(d, add_special_tokens=False)
    return out


@torch.no_grad()
def measure_stage_via_logprob_local(
    model_name: str,
    history_text: str,
    marked_speaker: str,
    marked_turn_text: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """Run a forward pass locally, softmax over the 6 digit-token positions."""
    try:
        model, tok = _load_local(model_name, device)
        user = _build_user(history_text, marked_speaker, marked_turn_text)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user},
        ]
        # Some chat templates reject the system role (Gemma); fold if needed.
        # Qwen3 enables thinking mode by default — pass enable_thinking=False
        # so the next token is a digit, not <think>.
        is_qwen3 = "qwen3" in model_name.lower()
        ct_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if is_qwen3:
            ct_kwargs["enable_thinking"] = False
        try:
            prompt_ids = tok.apply_chat_template(messages, **ct_kwargs)
        except Exception:
            merged = JUDGE_SYSTEM + "\n\n" + user
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": merged}], **ct_kwargs,
            )
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = prompt_ids.to(device)

        # Forward to get next-token logits at the end of the prompt
        out = model(input_ids=prompt_ids, use_cache=False, return_dict=True)
        next_logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(next_logits, dim=-1)

        digit_ids = _digit_token_ids(tok)
        raw: dict[str, float] = {}
        per_digit_logp = []
        for d in "123456":
            ids = digit_ids[d]
            # Use logprob of the first token (we assume single-token digit;
            # if multi-token, this still gives a meaningful relative ordering).
            lp = float(log_probs[ids[0]].item())
            raw[d] = lp
            per_digit_logp.append(lp)

        # Softmax-normalize over the 6 digit logprobs to get a proper
        # distribution. (We don't want masses below floor to leak in.)
        max_lp = max(per_digit_logp)
        exps = [math.exp(lp - max_lp) for lp in per_digit_logp]
        Z = sum(exps)
        dist = [e / Z for e in exps]

        argmax_idx = max(range(6), key=lambda i: dist[i])
        max_prob = dist[argmax_idx]
        entropy = -sum(p * math.log(p) for p in dist if p > 1e-12)

        return {
            "distribution":  dist,
            "argmax_idx":    argmax_idx,
            "argmax_stage":  STAGE_ORDER[argmax_idx],
            "max_prob":      max_prob,
            "entropy":       entropy,
            "raw_logprobs":  raw,
            "error":         None,
        }
    except Exception as e:
        return {
            "distribution":  None,
            "argmax_idx":    None,
            "argmax_stage":  None,
            "max_prob":      None,
            "entropy":       None,
            "raw_logprobs":  None,
            "error":         repr(e),
        }
