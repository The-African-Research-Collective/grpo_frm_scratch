import torch
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional


@dataclass
class MiniBatch:
    input_prompts: List[str]
    expected_outputs: List[str]
    inputs: List[str]


@dataclass
class Reward:
    name: str
    weight: float
    fn: Callable


@dataclass
class Group:
    input_text: str
    ground_truth: str
    responses: List[str]
    rewards: Dict[str, List[float]]

    # tensors (CPU)
    prompt_ids: Optional[torch.Tensor] = None  # (R, P_pad)
    completion_ids: Optional[torch.Tensor] = None  # (R, C_pad)
    prompt_completion_ids: Optional[torch.Tensor] = None  # (R, P_pad + C_pad)
    attention_mask: Optional[torch.Tensor] = None  # (R, P_pad + C_pad)
    completion_mask: Optional[torch.Tensor] = None  # (R, C_pad) 1 for completion tokens
    sampling_per_token_logps: Optional[torch.Tensor] = None  # (R, C_pad) log p under rollout policy

    advantages: Optional[torch.Tensor] = None  # (R,)
