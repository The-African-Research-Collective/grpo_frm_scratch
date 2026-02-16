import yaml
import time
import argparse
import logging
import pprint
import torch
import wandb

import numpy as np
from datetime import timedelta

from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate.utils import InitProcessGroupKwargs, DataLoaderConfiguration
from transformers import AutoTokenizer, AutoModelForCausalLM

from rewards import structure_reward, translation_reward
from data_class import Reward
from datasets_classes import AfriDocMTDataset
from model_utils import generate_grpo_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REWARD_FUNCTIONS = {
    "translation_reward": translation_reward,
    "structure_reward": structure_reward,
}

DATASET_CLASSES = {
    "afridoc_mt": AfriDocMTDataset,
}


@torch.no_grad()
def evaluate(
    accelerator,
    model,
    eval_dataloader,
    rewards,  # List[Reward]
    tokenizer,
    global_step: int,
    num_eval_rollouts: int = 1,
    max_generation_length: int = 4096,
    log_rollouts_table: bool = True,
    table_name: str = "eval/rollouts_table",
):
    accelerator.print("Starting evaluation...")
    model.eval()

    reward_names = [r.name for r in rewards]

    # Local accumulation (per process)
    local_sum = {rn: 0.0 for rn in reward_names}
    local_count = {rn: 0 for rn in reward_names}

    # We'll gather table rows as Python objects (strings + floats)
    local_rows = []  # each element: dict with all row fields

    # A stable unique id even if you don't have example IDs in the batch
    # (better: if your dataset provides example_id, use it instead)
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
            # Make a unique example id (process, local counter)
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

    # ----------------------------
    # Aggregate scalar eval metrics (distributed)
    # ----------------------------
    # Turn sums/counts into tensors and reduce across processes
    sum_tensor = torch.tensor([local_sum[rn] for rn in reward_names], device=accelerator.device, dtype=torch.float64)
    cnt_tensor = torch.tensor([local_count[rn] for rn in reward_names], device=accelerator.device, dtype=torch.float64)

    sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
    cnt_tensor = accelerator.reduce(cnt_tensor, reduction="sum")

    # Only main process computes final means + logs
    log_payload = {}
    results = defaultdict(list)  # keep your return type; we'll also fill it if you want

    if accelerator.is_main_process:
        means = (sum_tensor / torch.clamp(cnt_tensor, min=1.0)).cpu().numpy().tolist()
        for rn, avg_reward in zip(reward_names, means):
            avg_reward = float(avg_reward)
            accelerator.print(f"Average {rn}: {avg_reward}")
            log_payload[f"eval/{rn}"] = avg_reward

    # ----------------------------
    # Gather ALL rollouts for W&B table (distributed)
    # ----------------------------
    if log_rollouts_table:
        gathered = gather_object(local_rows)  # list-of-lists (one per process)
        if accelerator.is_main_process:
            columns = ["example_uid", "input_text", "ground_truth", "rollout_idx", "response"]
            columns += [f"reward/{rn}" for rn in reward_names]
            columns += ["reward/total"]

            table = wandb.Table(columns=columns)
            for r in gathered:
                row = [
                    r["example_uid"],
                    r["input_text"],
                    r["ground_truth"],
                    r["rollout_idx"],
                    r["response"],
                ]
                row += [r[f"reward/{rn}"] for rn in reward_names]
                row += [r["reward/total"]]
                table.add_data(*row)

            log_payload[table_name] = table

    model.train()
    return results, log_payload


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration: {pprint.pformat(config)}")

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        log_with="wandb",
        dataloader_config=dataloader_config,
        kwargs_handlers=[timeout_kwargs],
    )
    device = accelerator.device

    rewards = [Reward(name=reward_name, weight=reward_config["weight"], fn=REWARD_FUNCTIONS[reward_name]) for reward_name, reward_config in config["rewards"].items()]
    logger.info(f"Initialized reward functions: {[reward.name for reward in rewards]}")

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["pretrained_model_path"],
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_path"])

    dataset_class = DATASET_CLASSES[config["train_dataset"]["dataset_class_name"]]

    # Load evaluation and train dataset
    train_dataset = dataset_class(
        dataset_name_or_path=config["train_dataset"]["dataset_name_or_path"],
        split=config["train_dataset"]["split"],
        subset=config["train_dataset"]["subset"],
        num_samples=config["train_dataset"]["num_samples"],
        tokenizer=tokenizer,
        source_languages=config["train_dataset"]["source_languages"],
        target_language=config["train_dataset"]["target_language"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_dataset"]["batch_size"],
        collate_fn=dataset_class.collate_fn,
    )

    logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")

    eval_dataset = dataset_class(
        dataset_name_or_path=config["eval_dataset"]["dataset_name_or_path"],
        split=config["eval_dataset"]["split"],
        subset=config["eval_dataset"]["subset"],
        num_samples=config["eval_dataset"]["num_samples"],
        tokenizer=tokenizer,
        source_languages=config["eval_dataset"]["source_languages"],
        target_language=config["eval_dataset"]["target_language"],
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["eval_dataset"]["batch_size"],
        collate_fn=dataset_class.collate_fn,
    )

    # Load optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["training"]["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=float(config["training"]["learning_rate"]),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    # model = torch.compile(model, mode="max-autotune")

    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples.")
    logger.info("Running evaluation before training...")

    wandb_cfg = config.get("wandb", {})
    accelerator.init_trackers(
        wandb_cfg.get("project", "grpo"),
        config,
        init_kwargs={"wandb": {"name": wandb_cfg.get("run_name", None), "entity": wandb_cfg.get("entity", None)}},
    )

    results, log_datatable = evaluate(
        accelerator=accelerator,
        model=model,
        eval_dataloader=eval_dataloader,
        rewards=rewards,
        tokenizer=tokenizer,
        global_step=0,
        max_generation_length=config["training"]["max_generation_length"],
        log_rollouts_table=True,
    )
    accelerator.log(log_datatable, step=0)

    start_time = time.time()
    global_step = 0

    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{config['training']['num_epochs']}...")
        model.train()

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            global_step += 1
            wandb_table_name = "train/rollouts_table"

            batch_group = generate_grpo_rollout(
                batch_data=batch,
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                num_rollouts=config["training"]["num_rollouts"],
                max_generation_length=config["training"]["max_generation_length"],
                reward_functions=rewards,
                device=device,
            )

            step_loss = []
            step_entropy = []
            step_rewards = []
            step_rewards_std = []

            log_payload = {}
            reward_names = [r.name for r in rewards]
            columns = [
                "example_idx",
                "input_text",
                "ground_truth",
                "rollout_idx",
                "response",
            ]
            # per-reward columns
            columns += [f"reward/{rn}" for rn in reward_names]
            columns += ["reward/total"]
            table = []
            table_index = 0

            # calculate advantages per group
            for group in batch_group:
                group_rewards = [sum(outs) for outs in zip(*group.rewards.values())]
                mean_rewards = np.mean(group_rewards)
                std_rewards = np.std(group_rewards) + 1e-4
                group.advantages = torch.tensor([(reward - mean_rewards) / std_rewards for reward in group_rewards]).to(device)

                combined_ids = torch.cat([group.input_ids, group.output_ids], dim=1)[:, :-1]
                attention_mask = (combined_ids != tokenizer.pad_token_id).long()

                entropy = 0.0

                with torch.autocast(device_type="cuda", dtype=getattr(model, "dtype", None)):
                    outputs = model(
                        input_ids=combined_ids,
                        attention_mask=attention_mask,
                    )
                    target_ids = combined_ids
                    outputs_logits = outputs.logits

                    log_probs = -torch.nn.functional.cross_entropy(
                        outputs_logits.view(-1, outputs_logits.size(-1)),
                        target_ids.reshape(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction="none",
                    ).view(target_ids.size())

                    with torch.no_grad():
                        group_token_entropy = compute_entropy(outputs_logits)
                        entropy = entropy + (group_token_entropy * attention_mask).sum() / attention_mask.sum()

                    advantages = group.advantages.unsqueeze(1).expand_as(log_probs)
                    loss = -(advantages * log_probs).sum() / attention_mask.sum()

                    accelerator.backward(loss)

                    step_loss.append(loss.detach().item())
                    step_entropy.append(entropy.detach().item())
                    step_rewards.append(float(mean_rewards))
                    step_rewards_std.append(float(std_rewards))

                # log rollouts to W&B Table
                if config["wandb"]["log_rollouts_table"]:
                    n_rollouts = len(group.responses)

                    for k in range(n_rollouts):
                        row = [table_index, group.input_text, group.ground_truth, k, group.responses[k]]

                        per_reward_vals = []
                        for rn in reward_names:
                            v = group.rewards[rn][k]
                            per_reward_vals.append(float(v))

                        total = float(sum(per_reward_vals))
                        row += per_reward_vals
                        row += [total]
                        table_index += 1
                        table.append(row)

            batch_loss = torch.tensor(step_loss, device=device).mean()
            batch_entropy = torch.tensor(step_entropy, device=device).mean()
            batch_reward_mean = torch.tensor(step_rewards, device=device).mean()
            batch_reward_std = torch.tensor(step_rewards_std, device=device).mean()

            # update the policy
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            rows = gather_object(table)  # list-of-lists (one per process)
            if accelerator.is_main_process:
                table = wandb.Table(columns=columns, data=[part for part in rows])
                log_payload[wandb_table_name] = table
                accelerator.log(log_payload, step=global_step)

                payload = {
                    "train/loss": accelerator.gather(batch_loss).mean().item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/entropy": accelerator.gather(batch_entropy).mean().item(),
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                    "train/time_elapsed_sec": float(time.time() - start_time),
                    "train/reward_mean": accelerator.gather(batch_reward_mean).mean().item(),
                    "train/reward_std": accelerator.gather(batch_reward_std).mean().item(),
                }
                accelerator.log(payload, step=global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
