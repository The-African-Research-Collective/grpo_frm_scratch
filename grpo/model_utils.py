from transformers import AutoModel, AutoTokenizer
from grpo.data_class import MiniBatch

def generate_grpo_rollout(batch_data: MiniBatch,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        num_generations: int,
        max_generation_length: int ):

    