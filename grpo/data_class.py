import torch
from dataclasses import dataclass
from typing import List, Dict, Callable


@dataclass
class MiniBatch:
    input_prompts: List[str]
    expected_outputs: List[str]
    inputs: List[str]


@dataclass
class Group:
    input_text: str
    ground_truth: str
    responses: List[str]
    rewards: Dict[str, List]
    advantages: List[float] = None
    output_ids: torch.Tensor = None
    input_ids: torch.Tensor = None


@dataclass
class Reward:
    name: str
    weight: float
    fn: Callable
