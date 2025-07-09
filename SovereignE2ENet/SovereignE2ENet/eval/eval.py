from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Simple perplexity-style check
prompt = "The sovereiprompt = "The sovereie = "mps" if torch.backends.mps.is_available() else "cpu"
model = Autmodel = Autmodel = Aom_pretrained("sshleifer/tiny-gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")
