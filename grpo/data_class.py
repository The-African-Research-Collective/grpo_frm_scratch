from dataclasses import dataclass

@dataclass
class MiniBatch:
    input_ids: List[List[int]]
    input_tokens: List[List[str]]
    input_prompts: List[str]
    expected_outputs: List[str]
    inputs: List[str]