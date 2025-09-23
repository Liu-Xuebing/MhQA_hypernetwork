from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/disk/liuxb/code/Multi-EMoE/Decomposer/falcon3-1b-musique")
model = AutoModelForCausalLM.from_pretrained("/disk/liuxb/code/Multi-EMoE/Decomposer/falcon3-1b-musique",
                                             device_map="auto")


prompt = ("Decompose the following question into sub-questions:\n"
          "How long had Pfrang Association's headquarters location been the capitol city of Yaxing Coach's headquarters location?\n")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id)

print(outputs)

output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)
output = output.split('sub-answer:')[0]

# prompt =output + 'sub-answer: Michael Jace'
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=64)
# output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(output)
# print(output.split('Sub-questions: ')[1].split('\n'))

