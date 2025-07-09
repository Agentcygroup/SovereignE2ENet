import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="sshleifer/tiny-gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Example usage
model, tokenizer = load_model("gpt2")
print(f"Model {model.config._name_or_path} loaded successfully.")
