# -*- coding: utf-8 -*-
"""
Image generation generator
"""
import torch
import math
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk


@torch.no_grad()
def generate_image(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits = model(uncond, infer=True).logits[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
        else:
            logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


