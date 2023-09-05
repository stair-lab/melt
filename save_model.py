import torch
from peft import AutoPeftModelForCausalLM
from script_arguments import ScriptArguments
from transformers import AutoTokenizer

args = ScriptArguments()

model_name = args.new_model
new_model_name = f"{model_name}-merged"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained(new_model_name)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.save_pretrained(new_model_name)
