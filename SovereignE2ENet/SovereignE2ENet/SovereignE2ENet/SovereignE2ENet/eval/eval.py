import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from nltk.translate.bleu_score import sentence_bleu

# Load model and tokenizer
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("models/checkpoint_epoch_3").to(device)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# Load the dataset for evaluation
dataset = load_dataset('json', data_files='data/real_world_instructions.json', split='train')

# Perplexity Evaluation
model.eval()
perplexity = 0
for example in dataset:
    input_ids = tokenizer(example['instruction'], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity += math.exp(loss.item())

perplexity /= len(dataset)
print(f"Perplexity: {perplexity:.4f}")

# BLEU Evaluation (optional)
bleu_scores = []
for example in dataset:
    generated = tokenizer.decode(model.generate(tokenizer(example['instruction'], return_tensors="pt").input_ids.to(device))[0], skip_special_tokens=True)
    reference = [example['response'].split()]
    hypothesis = generated.split()
    bleu_scores.append(sentence_bleu(reference, hypothesis))

average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {average_bleu:.4f}")
