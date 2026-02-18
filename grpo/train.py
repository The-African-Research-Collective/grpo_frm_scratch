"""
Single-file GRPO training script (Accelerate + Transformers + PyTorch only)
Loss matches HuggingFace TRL GRPOTrainer logic (clipped surrogate + optional KL penalty).

Usage:
  accelerate launch grpo_single_file.py --config path/to/config.yaml

Notes:
- This script expects your dataset class AfriDocMTDataset, reward fns, and batch format
  to be compatible with how you were using them.
- It computes "sampling_per_token_logps" by running a forward pass on the generated
  prompt+completion under the *same model at rollout time*. This matches TRL's need for
  sampling logps (i.e., log-probs under the rollout policy).
"""

import argparse
import logging
import pprint
import time
from datetime import timedelta
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, InitProcessGroupKwargs, gather_object
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from rewards import structure_reward, translation_reward
from datasets_classes import AfriDocMTDataset
from data_class import Group, Reward
from model_utils import generate_grpo_rollout


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# Reward + Dataset registries
# ----------------------------
REWARD_FUNCTIONS = {
    "translation_reward": translation_reward,
    "structure_reward": structure_reward,
}

DATASET_CLASSES = {
    "afridoc_mt": AfriDocMTDataset,
}


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) token ids
    returns: (B, T) log p(label_t | ...)
    """
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


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

    # Forward current policy
    out = model(input_ids=pc_ids, attention_mask=attn, use_cache=False)
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


# ----------------------------
# Evaluation (optional W&B table)
# ----------------------------
@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    rewards: List[Reward],
    tokenizer: AutoTokenizer,
    global_step: int,
    num_eval_rollouts: int = 1,
    max_generation_length: int = 512,
    log_rollouts_table: bool = True,
    table_name: str = "eval/rollouts_table",
):
    accelerator.print("Starting evaluation...")
    model.eval()

    reward_names = [r.name for r in rewards]
    local_sum = {rn: 0.0 for rn in reward_names}
    local_count = {rn: 0 for rn in reward_names}
    local_rows = []
    local_example_idx = 0
    proc = accelerator.process_index

    gen_model = accelerator.unwrap_model(model)

    for batch in eval_dataloader:
        batch_group = generate_grpo_rollout(
            batch_data=batch,
            model=gen_model,
            tokenizer=tokenizer,
            num_rollouts=num_eval_rollouts,
            max_generation_length=max_generation_length,
            reward_functions=rewards,
            device=accelerator.device,
            temperature=0.0,
        )

        for group in batch_group:
            ex_uid = f"p{proc}-ex{local_example_idx}"
            local_example_idx += 1

            n_rollouts = len(group.responses)
            for k in range(n_rollouts):
                per_reward_vals = []
                for rn in reward_names:
                    v = float(group.rewards[rn][k])
                    per_reward_vals.append(v)
                    local_sum[rn] += v
                    local_count[rn] += 1

                total = float(sum(per_reward_vals))
                if log_rollouts_table:
                    local_rows.append(
                        {
                            "example_uid": ex_uid,
                            "input_text": group.input_text,
                            "ground_truth": group.ground_truth,
                            "rollout_idx": k,
                            "response": group.responses[k],
                            **{f"reward/{rn}": per_reward_vals[i] for i, rn in enumerate(reward_names)},
                            "reward/total": total,
                        }
                    )

    # reduce sums/counts
    sum_tensor = torch.tensor([local_sum[rn] for rn in reward_names], device=accelerator.device, dtype=torch.float64)
    cnt_tensor = torch.tensor([local_count[rn] for rn in reward_names], device=accelerator.device, dtype=torch.float64)
    sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
    cnt_tensor = accelerator.reduce(cnt_tensor, reduction="sum")

    log_payload = {}
    if accelerator.is_main_process:
        means = (sum_tensor / torch.clamp(cnt_tensor, min=1.0)).cpu().numpy().tolist()
        for rn, avg_reward in zip(reward_names, means):
            log_payload[f"eval/{rn}"] = float(avg_reward)

    if log_rollouts_table:
        gathered = gather_object(local_rows)  # list of rows from each proc
        if accelerator.is_main_process:
            columns = ["example_uid", "input_text", "ground_truth", "rollout_idx", "response"]
            columns += [f"reward/{rn}" for rn in reward_names]
            columns += ["reward/total"]
            table = wandb.Table(columns=columns)
            for r in gathered:
                table.add_data(
                    r["example_uid"],
                    r["input_text"],
                    r["ground_truth"],
                    r["rollout_idx"],
                    r["response"],
                    *[r[f"reward/{rn}"] for rn in reward_names],
                    r["reward/total"],
                )
            log_payload[table_name] = table

    model.train()
    return log_payload


# ----------------------------
# Main
# ----------------------------
def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration:\n{pprint.pformat(config)}")

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        log_with="wandb",
        dataloader_config=dataloader_config,
        kwargs_handlers=[timeout_kwargs],
    )
    device = accelerator.device

    # rewards
    rewards = [Reward(name=reward_name, weight=reward_cfg["weight"], fn=REWARD_FUNCTIONS[reward_name]) for reward_name, reward_cfg in config["rewards"].items()]
    logger.info(f"Initialized reward functions: {[r.name for r in rewards]}")

    # model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["pretrained_model_path"],
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # reference model (optional KL)
    beta = float(config["training"].get("beta", 0.0))
    ref_model = None
    if beta != 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config["model"]["pretrained_model_path"],
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # datasets/dataloaders
    dataset_class = DATASET_CLASSES[config["train_dataset"]["dataset_class_name"]]

    train_dataset = dataset_class(
        dataset_name_or_path=config["train_dataset"]["dataset_name_or_path"],
        split=config["train_dataset"]["split"],
        subset=config["train_dataset"]["subset"],
        num_samples=config["train_dataset"]["num_samples"],
        tokenizer=tokenizer,
        source_languages=config["train_dataset"]["source_languages"],
        target_language=config["train_dataset"]["target_language"],
    )
    eval_dataset = dataset_class(
        dataset_name_or_path=config["eval_dataset"]["dataset_name_or_path"],
        split=config["eval_dataset"]["split"],
        subset=config["eval_dataset"]["subset"],
        num_samples=config["eval_dataset"]["num_samples"],
        tokenizer=tokenizer,
        source_languages=config["eval_dataset"]["source_languages"],
        target_language=config["eval_dataset"]["target_language"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_dataset"]["batch_size"],
        collate_fn=dataset_class.collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["eval_dataset"]["batch_size"],
        collate_fn=dataset_class.collate_fn,
        shuffle=False,
    )

    # optimizer
    no_decay = ["bias", "layer_norm.weight", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(config["training"]["weight_decay"]),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(config["training"]["learning_rate"]))

    # prepare
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # wandb init
    wandb_cfg = config.get("wandb", {})
    accelerator.init_trackers(
        wandb_cfg.get("project", "grpo"),
        config,
        init_kwargs={"wandb": {"name": wandb_cfg.get("run_name", None), "entity": wandb_cfg.get("entity", None)}},
    )

    # training params
    epsilon_low = float(config["training"].get("epsilon_low", 0.2))
    epsilon_high = float(config["training"].get("epsilon_high", 0.2))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    num_rollouts = int(config["training"]["num_rollouts"])
    max_gen_len = int(config["training"]["max_generation_length"])
    num_epochs = int(config["training"]["num_epochs"])
    log_rollouts_table = bool(config.get("wandb", {}).get("log_rollouts_table", False))

    # Run eval before trainining
    eval_logs = evaluate(
        accelerator=accelerator,
        model=model,
        eval_dataloader=eval_dataloader,
        rewards=rewards,
        tokenizer=tokenizer,
        global_step=0,
    )
    accelerator.log(eval_logs, step=0)

    start_time = time.time()
    global_step = 0

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        model.train()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            global_step += 1

            # 1) Generate rollouts (each process does its own shard)
            batch_group = generate_grpo_rollout(
                batch_data=batch,
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                num_rollouts=num_rollouts,
                max_generation_length=max_gen_len,
                reward_functions=rewards,
                device=accelerator.device,
                temperature=float(config["training"].get("temperature", 0.7)),
            )

            accelerator.wait_for_everyone()

            step_losses = []
            step_rewards_mean = []
            step_rewards_std = []

            # for W&B table
            reward_names = [r.name for r in rewards]
            columns = ["example_idx", "input_text", "ground_truth", "rollout_idx", "response"]
            columns += [f"reward/{rn}" for rn in reward_names]
            columns += ["reward/total"]
            table_rows = []
            table_index = 0

            with accelerator.accumulate(model):
                for group in batch_group:
                    # advantages per group (normalized within group)
                    group_rewards = [sum(outs) for outs in zip(*group.rewards.values())]
                    mean_r = float(np.mean(group_rewards))
                    std_r = float(np.std(group_rewards) + 1e-4)
                    advantages = torch.tensor([(r - mean_r) / std_r for r in group_rewards], dtype=torch.float32)
                    group.advantages = advantages  # store on CPU

                    loss = grpo_loss(
                        model=model,
                        ref_model=ref_model,
                        group=group,
                        advantages=advantages,
                        beta=beta,
                        epsilon_low=epsilon_low,
                        epsilon_high=epsilon_high,
                        device=accelerator.device,
                    )

                    accelerator.backward(loss)
                    step_losses.append(loss.detach().float().item())
                    step_rewards_mean.append(mean_r)
                    step_rewards_std.append(std_r)

                    # table
                    if log_rollouts_table:
                        n_rollouts = len(group.responses)
                        for k in range(n_rollouts):
                            per_reward_vals = [float(group.rewards[rn][k]) for rn in reward_names]
                            total = float(sum(per_reward_vals))
                            row = [
                                table_index,
                                group.input_text,
                                group.ground_truth,
                                k,
                                group.responses[k],
                                *per_reward_vals,
                                total,
                            ]
                            table_rows.append(row)
                            table_index += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # reduce metrics
            batch_loss = torch.tensor(step_losses, device=device).mean()
            batch_reward_mean = torch.tensor(step_rewards_mean, device=device).mean()
            batch_reward_std = torch.tensor(step_rewards_std, device=device).mean()

            loss_g = accelerator.gather(batch_loss).mean().item()
            rew_mg = accelerator.gather(batch_reward_mean).mean().item()
            rew_sg = accelerator.gather(batch_reward_std).mean().item()

            log_payload = {
                "train/loss": loss_g,
                "train/epoch": epoch + 1,
                "train/step": global_step,
                "train/time_elapsed_sec": float(time.time() - start_time),
                "train/reward_mean": rew_mg,
                "train/reward_std": rew_sg,
            }

            if log_rollouts_table:
                rows = gather_object(table_rows)  # list-of-lists across procs
                if accelerator.is_main_process:
                    table = wandb.Table(columns=columns, data=[r for r in rows])
                    log_payload["train/rollouts_table"] = table

            accelerator.log(log_payload, step=global_step)

        # Optional: eval each epoch
        if config["training"].get("eval_each_epoch", False):
            eval_logs = evaluate(
                accelerator=accelerator,
                model=model,
                eval_dataloader=eval_dataloader,
                rewards=rewards,
                tokenizer=tokenizer,
                global_step=global_step,
                num_eval_rollouts=int(config["eval_dataset"].get("num_eval_rollouts", 1)),
                max_generation_length=int(config["eval_dataset"].get("max_generation_length", 256)),
                log_rollouts_table=bool(config.get("wandb", {}).get("log_eval_rollouts_table", False)),
                table_name="eval/rollouts_table",
            )
            accelerator.log(eval_logs, step=global_step)

    accelerator.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
