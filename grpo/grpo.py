import torch

import torch.nn.functional as F

from typing import Optional
from data_class import Group 

def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) token ids
    returns: (B, T) log p(label_t | ...)
    """
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)



def grpo_loss(
    model: torch.nn.Module,
    ref_model: Optional[torch.nn.Module],
    group: Group,
    advantages: torch.Tensor,  # (R,)
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Matches TRL's GRPOTrainer "grpo" loss branch:
      - importance ratio: exp(cur_logps - sampling_logps)
      - clipped ratio
      - per-token objective: -min(r*A, clip(r)*A)
      - optional KL penalty: exp(ref-cur) - (ref-cur) - 1, multiplied by beta
      - masked mean over completion tokens
    """

    pc_ids = group.prompt_completion_ids.to(device)  # (R, P+C)
    attn = group.attention_mask.to(device)  # (R, P+C)
    cmask = group.completion_mask.to(device).float()  # (R, C)
    sampling_logps = group.sampling_per_token_logps.to(device)  # (R, C)
    
    token_type_ids = torch.zeros_like(pc_ids).to(device)

    # Forward current policy
    out = model(input_ids=pc_ids, attention_mask=attn, use_cache=False, token_type_ids=token_type_ids)
    logits = out.logits  # (R, P+C, V)

    # next-token alignment
    labels = pc_ids[:, 1:]  # (R, P+C-1)
    logits = logits[:, :-1, :]  # (R, P+C-1, V)

    C = cmask.size(1)
    comp_labels = labels[:, -C:]  # (R, C)
    comp_logits = logits[:, -C:, :]  # (R, C, V)

    per_token_logps = selective_log_softmax(comp_logits, comp_labels) * cmask  # (R, C)

    # Importance ratio
    log_iw = per_token_logps - sampling_logps
    coef_1 = torch.exp(log_iw)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    adv = advantages.to(device).float().unsqueeze(1)  # (R, 1)
    per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)  # (R, C)

    # Optional KL penalty against reference
    if ref_model is not None and beta != 0.0:
        with torch.no_grad():
            ref_out = ref_model(input_ids=pc_ids, attention_mask=attn, use_cache=False)
            ref_logits = ref_out.logits[:, :-1, :]  # align
            ref_comp_logits = ref_logits[:, -C:, :]
            ref_per_token_logps = selective_log_softmax(ref_comp_logits, comp_labels) * cmask

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl

    # masked mean over completion tokens
    denom = cmask.sum(dim=1).clamp(min=1.0)
    loss = ((per_token_loss * cmask).sum(dim=1) / denom).mean()
    return loss