from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

@app.get("/generate")
def generate(prompt: str = "Hello, sovereign AI"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    return {"output": tokenizer.decode(outputs[0], skip_special_tokens=True)}
