# pip install -q transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Prompt
prompt = "//module half adder "
device='cuda'
# Load model and tokenizer
model_name = "shailja/fine-tuned-codegen-6B-Verilog"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Sample
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
sample = model.generate(input_ids, max_length=128, temperature=0.5, top_p=0.9)

print(tokenizer.decode(sample[0], truncate_before_pattern=[r"endmodule"]) + "endmodule")
