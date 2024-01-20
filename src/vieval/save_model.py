import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = sys.argv[1]
new_model_name = sys.argv[2]
if len(sys.argv) > 3:
    push = sys.argv[3]
else:
    push = None

model = AutoPeftModelForCausalLM.from_pretrained(
    model_name, device_map={"": 0}, torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained(new_model_name)
if push == "push":
    model.push_to_hub(
        new_model_name,
        commit_message="Init model",
        private=True,
        use_auth_token=True,
        safe_serialization=True,
    )


tokenizer_map = {
    "7b": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-chat-hf",
    "70b": "meta-llama/Llama-2-70b-chat-hf",
}
tokenizer_name = None
for size in tokenizer_map:
    if size in model_name:
        tokenizer_name = tokenizer_map[size]
        break
if tokenizer_name is None:
    raise ValueError("No tokenizer found for model name")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.save_pretrained(new_model_name)
if push == "push":
    tokenizer.push_to_hub(
        new_model_name,
        commit_message="Init tokenizer",
        private=True,
        use_auth_token=True,
    )
