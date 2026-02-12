import evaluate
import pandas as pd
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HfDataset
from datasets import load_dataset
from jinja2 import Environment
from typing import Dict, List

from grpo.data_class import MiniBatch

SYSTEM_PROMPT = """You are a helpful assistant for translating documents for low-resource languages
You first think about the reasoning process to translate the document, and then provide the final answer
"""

USER_PROMPT = """You are given a document in {source_language}. Your task is to translate the document into {target_language}.
Please follow these steps to complete the translation:
- Read the entire document carefully to understand its context and meaning.
- Identify any cultural references or idiomatic expressions that may need special attention during translation.
- Show your thought process in <think> </think> tags.
- And return the final answer in <response> </response> tags

Here is the document to be translated:
{source_text}

Here is how i would translate this text\n<think>
"""

MAPPING = {
    "sw": "Swahili",
    "am": "Amharic",
    "yo": "Yoruba",
    "ha": "Hausa",
    "zu": "Zulu",
    "en": "English"
}


class AfriDocMTDataset(Dataset):
    def __init__(self, dataset_name_or_path: str, split: str, subset:str, num_samples: str, tokenizer: AutoTokenizer, source_languages: list, target_language: str):

        self.tokenizer = tokenizer
        self.data = self.load_dataset_hgf(dataset_name_or_path, split, subset, num_samples, source_languages, target_language)

        self.chat_template = Environment().from_string(
            self.tokenizer.chat_template
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source_language = item['source_language']
        target_language = item['target_language']
        source_text = item['source_text']
        target_text = item['target_text']

        user_prompt = USER_PROMPT.format(source_language=source_language, target_language=target_language, source_text=source_text)
        print(user_prompt)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        input_prompt = self.chat_template.render(messages=messages, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(input_prompt)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "input_tokens": input_tokens,
            "input_prompt": input_prompt,
            "expected_output": target_text,
            "input": source_text
        }

    def load_dataset_hgf(self, dataset_name_or_path: str, split: str, subset:str, num_samples: str, source_languages: list, target_language: str):
        all_data = load_dataset(dataset_name_or_path, subset)[split]
        df = pd.DataFrame(all_data)

        total = []

        for lang in source_languages:
            trimmed_df = df[[lang, target_language]]
            trimmed_df = trimmed_df.rename(columns={lang: "source_text", target_language: "target_text"})
            trimmed_df["source_language"] = MAPPING[lang]
            trimmed_df["target_language"] = MAPPING[target_language]

            total.append(trimmed_df)
        
        merged_df = pd.concat(total, ignore_index=True)
        selected_dataset = HfDataset.from_pandas(merged_df)

        if num_samples == -1:
            return selected_dataset
        else:
            return selected_dataset.shuffle(seed=42).select(range(num_samples))

def collate_fn(batch: List[Dict]) -> MiniBatch:
    input_ids = [item["input_ids"] for item in batch]
    input_tokens = [item["input_tokens"] for item in batch]
    input_prompts = [item["input_prompt"] for item in batch]
    expected_outputs = [item["expected_output"] for item in batch]
    inputs = [item["input"] for item in batch]

    return MiniBatch(
        input_ids=input_ids,
        input_tokens=input_tokens,
        input_prompts=input_prompts,
        expected_outputs=expected_outputs,
        inputs=inputs
    )

if __name__ == "__main__":
    dataset = AfriDocMTDataset(
        dataset_name_or_path="masakhane/AfriDocMT",
        split="train",
        subset="doc_health_5",
        num_samples=100,
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"),
        source_languages=["yo"],
        target_language="en"
    )

    print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break