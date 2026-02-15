import yaml
import time
import argparse
import logging
import pprint
import torch
import wandb
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from grpo.rewards import structure_reward, translation_reward
from grpo.data_class import Reward
from grpo.datasets_classes import AfriDocMTDataset
from grpo.model_utils import generate_grpo_rollout

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
    model,
    eval_dataloader,
    rewards,  # List[Reward]
    tokenizer,
    device,
    dtype,
    global_step: int,
    num_eval_rollouts: int = 1,
    max_generation_length: int = 4096,
    log_rollouts_table: bool = True,
    table_name: str = "eval/rollouts_table",
):
    logger.info("Starting evaluation...")
    model.eval()

    group_list = []
    for batch in tqdm(eval_dataloader, desc="Running evaluation ..."):
        batch_group = generate_grpo_rollout(
            batch_data=batch,
            model=model,
            tokenizer=tokenizer,
            num_rollouts=num_eval_rollouts,
            max_generation_length=max_generation_length,
            reward_functions=rewards,
            device=model.device,
            dtype=dtype,
            temperature=0.0,  # deterministic generation for evaluation
        )
        group_list.extend(batch_group)

    # ----------------------------
    # Aggregate scalar eval metrics
    # ----------------------------
    results = defaultdict(list)
    for reward in rewards:
        for group in group_list:
            # group.rewards[reward.name] is a list length == num_eval_rollouts
            results[reward.name].extend(group.rewards[reward.name])

    log_payload = {}
    for reward in rewards:
        vals = results[reward.name]
        avg_reward = float(np.mean(vals)) if len(vals) else 0.0
        logger.info(f"Average {reward.name}: {avg_reward}")
        log_payload[f"eval/{reward.name}"] = avg_reward

    # ----------------------------
    # W&B Table: log ALL rollouts
    # One row per rollout
    # ----------------------------
    if log_rollouts_table:
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

        table = wandb.Table(columns=columns)

        for ex_i, group in enumerate(group_list):
            n_rollouts = len(group.responses)

            for k in range(n_rollouts):
                row = [
                    ex_i,
                    group.input_text,
                    group.ground_truth,
                    k,
                    group.responses[k],
                ]

                per_reward_vals = []
                for rn in reward_names:
                    # group.rewards[rn] is list[float] per rollout
                    v = group.rewards[rn][k]
                    per_reward_vals.append(float(v))

                total = float(sum(per_reward_vals))

                row += per_reward_vals
                row += [total]

                table.add_data(*row)

        # log the table + scalar metrics together
        log_payload[table_name] = table

    if log_payload:
        wandb.log(log_payload, step=global_step)

    return results


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def _init_wandb(config: dict):
    wandb_cfg = config.get("wandb", {})

    project = wandb_cfg.get("project", "grpo")
    run_name = wandb_cfg.get("run_name", None)
    entity = wandb_cfg.get("entity", None)

    # If user sets wandb.enabled: false, skip
    if wandb_cfg.get("enabled", True) is False:
        logger.info("wandb.enabled is False — skipping W&B init.")
        return False

    wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=config,
    )
    return True


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration: {pprint.pformat(config)}")
    _wandb_enabled = _init_wandb(config)

    rewards = [
        Reward(
            name=reward_name,
            weight=reward_config["weight"],
            fn=REWARD_FUNCTIONS[reward_name],
        )
        for reward_name, reward_config in config["rewards"].items()
    ]
    logger.info(f"Initialized reward functions: {[reward.name for reward in rewards]}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        config["model"]["pretrained_model_path"],
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
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

    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples.")
    logger.info("Running evaluation before training...")
    evaluate(
        model=model,
        eval_dataloader=eval_dataloader,
        rewards=rewards,
        tokenizer=tokenizer,
        device=model.device,
        dtype=torch.bfloat16,
        global_step=0,
        max_generation_length=config["training"]["max_generation_length"],
        log_rollouts_table=True,
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

    start_time = time.time()
    global_step = 0

    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{config['training']['num_epochs']}...")
        model.train()

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            global_step += 1
            wandb_table_name = f"train/rollouts_table_{global_step}"

            batch_group = generate_grpo_rollout(
                batch_data=batch,
                model=model,
                tokenizer=tokenizer,
                num_rollouts=config["training"]["num_rollouts"],
                max_generation_length=config["training"]["max_generation_length"],
                reward_functions=rewards,
                device=model.device,
                dtype=torch.bfloat16,
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
            table = wandb.Table(columns=columns)
            table_index = 0

            # calculate advantages per group
            for group in batch_group:
                group_rewards = [sum(outs) for outs in zip(*group.rewards.values())]
                mean_rewards = np.mean(group_rewards)
                std_rewards = np.std(group_rewards) + 1e-4
                group.advantages = torch.tensor([(reward - mean_rewards) / std_rewards for reward in group_rewards]).to(model.device)

                combined_ids = torch.cat([group.input_ids, group.output_ids], dim=1)

                attention_mask = torch.cat(
                    [
                        torch.zeros_like(group.input_ids),
                        (group.output_ids != tokenizer.pad_token_id).long(),
                    ],
                    dim=1,
                )

                attention_mask = attention_mask[:, :-1]
                entropy = 0.0

                with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=combined_ids[:, :-1],
                        attention_mask=attention_mask,
                    )
                    target_ids = combined_ids[:, 1:]
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

                    loss.backward()

                    step_loss.append(loss.item())
                    step_entropy.append(entropy.item())
                    step_rewards.append(float(mean_rewards))
                    step_rewards_std.append(float(std_rewards))

                # log rollouts to W&B Table
                if config["wandb"]["log_rollouts_table"]:
                    n_rollouts = len(group.responses)

                    for k in range(n_rollouts):
                        row = [
                            table_index,
                            group.input_text,
                            group.ground_truth,
                            k,
                            group.responses[k],
                        ]

                        per_reward_vals = []
                        for rn in reward_names:
                            v = group.rewards[rn][k]
                            per_reward_vals.append(float(v))

                        total = float(sum(per_reward_vals))
                        row += per_reward_vals
                        row += [total]
                        table_index += 1
                        table.add_data(*row)

            log_payload[wandb_table_name] = table

            print(log_payload)
            wandb.log(log_payload, step=global_step)

            # update the policy
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            payload = {
                "train/loss": sum(step_loss) / len(step_loss) if step_loss else 0.0,
                "train/grad_norm": float(grad_norm.item()),
                "train/entropy": sum(step_entropy) / len(step_entropy) if step_entropy else 0.0,
                "train/epoch": epoch + 1,
                "train/step": global_step,
                "train/time_elapsed_sec": float(time.time() - start_time),
                "train/reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
                "train/reward_std": float(np.mean(step_rewards_std)) if step_rewards_std else 0.0,
            }
            wandb.log(payload, step=global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
