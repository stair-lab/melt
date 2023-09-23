import sys
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = sys.argv[1]
new_model_name = sys.argv[2]


model = AutoPeftModelForCausalLM.from_pretrained(
    model_name, device_map={'': 0}, torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained(new_model_name)
model.push_to_hub(
    new_model_name,
    commit_message="Init model",
    private=True,
    token=True,
    safe_serialization=True
)


tokenizer_map = {
    '7b': "meta/llama-2-7b-hf",
    '13b': "meta/llama-2-13b-hf",
    '70b': "meta/llama-2-70b-hf",
}
tokenizer_name = None
for size in tokenizer_map:
    if size in model_name:
        tokenizer_name = tokenizer_map[size]
        break
if tokenizer_name is None:
    raise ValueError("No tokenizer found for model name")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name, trust_remote_code=True)
tokenizer.save_pretrained(new_model_name)
model.push_to_hub(
    new_model_name,
    commit_message="Init tokenizer",
    private=True,
    token=True
)
