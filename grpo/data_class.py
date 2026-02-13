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
    rewards: List[Dict]


@dataclass
class Reward:
    name: str
    weight: float
    fn: Callable
