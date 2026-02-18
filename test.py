from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")
inputs = tokenizer(["The biggest states in the USA by land area:", "Foo"] * 5, return_tensors="pt", padding=True).to(model.device)

gen_out = model.generate(**inputs)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))
print(gen_out)

# Passing one or more stop strings will halt generation after those strings are emitted
# Note that generating with stop strings requires you to pass the tokenizer too
gen_out = model.generate(**inputs, stop_strings=["Texas"], tokenizer=tokenizer)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))
print(gen_out)
