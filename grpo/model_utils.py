import os
import torch
import shutil
from accelerate.logging import get_logger
from torch.nn import functional as F

from transformers import AutoTokenizer
from typing import List, Optional
from tqdm import tqdm

from data_class import MiniBatch, Group, Reward

logger = get_logger(__name__)


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) token ids
    returns: (B, T) log p(label_t | ...)
    """
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


# ----------------------------
# Rollout generation (also computes sampling_per_token_logps)
# ----------------------------
@torch.no_grad()
def generate_grpo_rollout(
    batch_data: MiniBatch,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    num_rollouts: int,
    max_generation_length: int,
    reward_functions: List[Reward],
    device: torch.device,
    temperature: float = 0.5,
) -> List[Group]:
    model.eval()
    groups: List[Group] = []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for sampl_id in tqdm(range(len(batch_data.inputs)), desc="Generating rollouts", leave=False):
        messages = [batch_data.input_prompts[sampl_id] for _ in range(num_rollouts)]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_generation_length,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature and temperature != 0.0:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature))
        else:
            gen_kwargs.update(dict(do_sample=False))

        generation = model.generate(**inputs, **gen_kwargs)  # (R, P+C_gen)

        # prompt lengths from attention_mask (left padding)
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        gen_list = generation.detach().tolist()

        # Extract completion suffix per rollout
        completion_ragged: List[List[int]] = []
        for i, seq in enumerate(gen_list):
            p = int(prompt_lens[i])
            completion_ragged.append(seq[-(len(seq) - p) :])

        max_c = max(len(x) for x in completion_ragged)
        R = len(completion_ragged)

        completion_ids = torch.full((R, max_c), tokenizer.pad_token_id, dtype=torch.long)
        completion_mask = torch.zeros((R, max_c), dtype=torch.long)

        for i, comp in enumerate(completion_ragged):
            L = len(comp)
            completion_ids[i, :L] = torch.tensor(comp, dtype=torch.long)
            completion_mask[i, :L] = 1

        prompt_ids = inputs["input_ids"].detach().cpu()  # (R, P_pad)
        attn_prompt = inputs["attention_mask"].detach().cpu()  # (R, P_pad)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (R, P_pad+max_c)
        attention_mask = torch.cat([attn_prompt, completion_mask], dim=1)  # (R, P_pad+max_c)

        # sampling_per_token_logps (under rollout policy = model at generation time)
        pc_ids = prompt_completion_ids.to(device)
        pc_mask = attention_mask.to(device)
        out = model(input_ids=pc_ids, attention_mask=pc_mask, use_cache=False)
        logits = out.logits  # (R, P_pad+max_c, V)

        labels = pc_ids[:, 1:]
        logits = logits[:, :-1, :]

        comp_labels = labels[:, -max_c:]
        comp_logits = logits[:, -max_c:, :]

        sampling_per_token_logps = selective_log_softmax(comp_logits, comp_labels)  # (R, max_c)
        sampling_per_token_logps = sampling_per_token_logps * completion_mask.to(device)

        responses = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        ground_truth = [batch_data.expected_outputs[sampl_id]] * num_rollouts
        rewards = {rf.name: (rf.fn(responses, ground_truth) * rf.weight) for rf in reward_functions}

        group = Group(
            input_text=batch_data.inputs[sampl_id],
            ground_truth=batch_data.expected_outputs[sampl_id],
            responses=responses,
            rewards=rewards,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            sampling_per_token_logps=sampling_per_token_logps.detach().cpu(),
        )
        groups.append(group)

        del inputs, generation, out
        torch.cuda.empty_cache()

    model.train()
    return groups

def get_last_checkpoint(folder: str, incomplete: bool = False) -> Optional[str]:
    content = os.listdir(folder)
    checkpoint_steps = [path for path in content if path.startswith("step_")]
    checkpoint_epochs = [path for path in content if path.startswith("epoch_")]
    if len(checkpoint_steps) > 0 and len(checkpoint_epochs) > 0:
        logger.info("Mixed step and epoch checkpoints found. Using step checkpoints.")
        checkpoints = checkpoint_steps
    elif len(checkpoint_steps) == 0:
        checkpoints = checkpoint_epochs
    else:
        checkpoints = checkpoint_steps
    if not incomplete:
        checkpoints = [
            path
            for path in checkpoints
            if os.path.exists(os.path.join(folder, path, "COMPLETED"))
        ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: x.split("_")[-1]))


def get_last_checkpoint_path(config, incomplete: bool = False) -> str:
    # if output already exists and user does not allow overwriting, resume from there.
    # otherwise, resume if the user specifies a checkpoint.
    # else, start from scratch.
    # if incomplete is true, include folders without "COMPLETE" in the folder.
    last_checkpoint_path = None
    output_dir = config["training"]["output_dir"]
    
    if (
        output_dir
        and os.path.isdir(output_dir)
        and not config["training"]["overwrite_output_dir"]
    ):
        last_checkpoint_path = get_last_checkpoint(
            output_dir, incomplete=incomplete
        )
        if last_checkpoint_path is None:
            logger.warning(
                "Output directory exists but no checkpoint found. Starting from scratch."
            )
    return last_checkpoint_path

def is_checkpoint_folder(dir: str, folder: str) -> bool:
    return (
        folder.startswith("step_") or folder.startswith("epoch_")
    ) and os.path.isdir(os.path.join(dir, folder))

def clean_last_n_checkpoints(output_dir: str, keep_last_n_checkpoints: int) -> None:
    # remove the last checkpoint to save space
    folders = [f for f in os.listdir(output_dir) if is_checkpoint_folder(output_dir, f)]
    # find the checkpoint with the largest step
    checkpoints = sorted(folders, key=lambda x: int(x.split("_")[-1]))
    if keep_last_n_checkpoints != -1 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[: len(checkpoints) - keep_last_n_checkpoints]:
            logger.info(f"Removing checkpoint {checkpoint}")
            shutil.rmtree(os.path.join(output_dir, checkpoint))
    logger.info("Remaining files:" + str(os.listdir(output_dir)))



if __name__ == "__main__":
    from datasets_classes import AfriDocMTDataset, collate_fn
    from rewards import structure_reward, translation_reward
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

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    structure_reward_fn = Reward(name="structure_reward", weight=1, fn=structure_reward)
    translation_reward_fn = Reward(name="translation_reward", weight=1, fn=translation_reward)

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
        )
        break
