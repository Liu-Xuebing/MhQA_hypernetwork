from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/disk/liuxb/code/Multi-EMoE/Decomposer/falcon3-1b-mquake")
model = AutoModelForCausalLM.from_pretrained("/disk/liuxb/code/Multi-EMoE/Decomposer/falcon3-1b-mquake",
                                             device_map="auto")


prompt = ("Decompose the following question into sub-questions:\n"
          "What is the capital city of the country where the founder of Mike Mignola's employer holds citizenship?\nSub-questions:")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)



output = tokenizer.decode(outputs[0], skip_special_tokens=True)
output = output.split('\n\n')[0]
print(output.split('Sub-questions: ')[1].split('\n'))