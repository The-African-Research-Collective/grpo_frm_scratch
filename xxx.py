from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, DynamicCache
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda", torch_dtype=torch.bfloat16
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [[
    {"role": "system","content": "You are a helpful assistant."},
    {"role": "user", "content": "Describe  yourself in detail."}
],[
    {"role": "user", "content": "What is going on in the city of toronto"}
]
]

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, padding=True,
    return_dict=True, return_tensors="pt"
)

print(inputs["input_ids"])

input_lengths = len(inputs["input_ids"][0])

import time

start = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    inputs = {k: v.cuda() for k, v in inputs.items()}
    past_key_values = DynamicCache(config=model.config)
    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False, past_key_values=past_key_values)

outs = tokenizer.batch_decode(generation, skip_special_tokens=True)

end = time.time()

for i in range(len(messages)):
    print(f"Input {i}:")
    print(generation[i])
    print(len(generation[i]))
    print(f"Output: {outs[i][input_lengths:]} \n\n\n\n")

print(f"Generation took {end - start:.2f} seconds")

    