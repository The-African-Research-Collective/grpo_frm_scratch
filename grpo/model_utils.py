import torch

from transformers import AutoModel, AutoTokenizer
from typing import List
from tqdm import tqdm

from grpo.data_class import MiniBatch, Group, Reward


def generate_grpo_rollout(
    batch_data: MiniBatch,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    num_rollouts: int,
    max_generation_length: int,
    reward_functions: List[Reward],
    device: torch.device,
    dtype: torch.dtype,
) -> List[Group]:
    model.eval()
    groups = []

    # 1. for each input in the batch, generate num_generations continuations using the model
    with torch.autocast(device_type=device.type, dtype=dtype):
        for sampl_id in tqdm(range(len(batch_data.inputs)), desc="Generating rollouts"):
            messages = [batch_data.input_prompts[sampl_id] for i in range(num_rollouts)]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_lengths = [len(input_id) for input_id in inputs["input_ids"]]

            generation = model.generate(
                **inputs,
                max_new_tokens=max_generation_length,
                temperature=0.5,
                do_sample=True,
                use_cache=True,
            )
            output_ids = [
                gen[input_lengths[i] :]
                for i, gen in enumerate(generation.detach().tolist())
            ]

            responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            ground_truth = [batch_data.expected_outputs[sampl_id]] * num_rollouts

            rewards = {
                reward.name: reward.fn(responses, ground_truth) * reward.weight
                for reward in reward_functions
            }

            group = Group(
                input_text=batch_data.inputs[sampl_id],
                ground_truth=batch_data.expected_outputs[sampl_id],
                responses=responses,
                rewards=rewards,
            )

            groups.append(group)

    return groups


if __name__ == "__main__":
    from grpo.afridoc_mt import AfriDocMTDataset, collate_fn
    from grpo.rewards import structure_reward, translation_reward
    from grpo.data_class import Reward
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    dataset = AfriDocMTDataset(
        dataset_name_or_path="masakhane/AfriDocMT",
        split="train",
        subset="doc_tech_5",
        num_samples=100,
        tokenizer=AutoTokenizer.from_pretrained("google/gemma-3-4b-it"),
        source_languages=["yo"],
        target_language="en",
    )

    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    structure_reward_fn = Reward(name="structure_reward", weight=1, fn=structure_reward)
    translation_reward_fn = Reward(
        name="translation_reward", weight=1, fn=translation_reward
    )

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        generate_grpo_rollout(
            batch_data=batch,
            model=model,
            tokenizer=tokenizer,
            num_rollouts=2,
            max_generation_length=4096,
            reward_functions=[structure_reward_fn, translation_reward_fn],
            device=model.device,
            dtype=torch.bfloat16,
        )
        break
