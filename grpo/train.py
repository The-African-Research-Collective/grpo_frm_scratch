import yaml
import argparse
import logging
import pprint
import torch
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from grpo.rewards import structure_reward, translation_reward
from grpo.data_class import Reward
from grpo.datasets_classes import AfriDocMTDataset
from grpo.model_utils import generate_grpo_rollout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REWARD_FUNCTIONS = {
    "translation_reward": translation_reward,
    "structure_reward": structure_reward,
}

DATASET_CLASSES = {
    "afridoc_mt": AfriDocMTDataset,
}


def evaluate(model, eval_dataloader, rewards, tokenizer, device, dtype):
    logger.info("Starting evaluation...")

    group_list = []
    for batch in tqdm(eval_dataloader, desc="Running evaluation"):
        batch_group = generate_grpo_rollout(
            batch_data=batch,
            model=model,
            tokenizer=tokenizer,
            num_rollouts=1,
            max_generation_length=4096,
            reward_functions=rewards,
            device=model.device,
            dtype=torch.bfloat16,
        )

        group_list.extend(batch_group)

    results = defaultdict(list)

    for reward in rewards:
        for group in group_list:
            results[reward.name].append(group.rewards[reward.name])

    print(results)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration: {pprint.pformat(config)}")

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
        torch_dtype=torch.bfloat16,
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
    _train_dataloader = DataLoader(
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
