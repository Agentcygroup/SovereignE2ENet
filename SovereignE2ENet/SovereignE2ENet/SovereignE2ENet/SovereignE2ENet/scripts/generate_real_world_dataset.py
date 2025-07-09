import requests
from bs4 import BeautifulSoup
import json

# Function to scrape Wikipedia
def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    
    data = []
    for para in paragraphs[:10]:  # Scrape the first 10 paragraphs
        text = para.get_text()
        data.append({"instruction": "Summarize this text", "response": text})
    
    return data

# Scrape Wikipedia page on "Artificial Intelligence"
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
dataset = scrape_wikipedia(url)

# Save the dataset as JSON
with open("data/real_world_instructions.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Dataset generatprint("Dataset generatpri 5. Create advanced training script with gradient accumulation anprint("Dataset geneOF > scripts/train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Check device compatibility
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load custom dataset
dataset = load_dataset('json', data_files='data/real_world_instructions.json', split='train')
def tokenize_function(examples):
    return tokenizer(examples['instruction'], truncation=True, padding='max_length', max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets.shuffle(seed=42)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
num_epochs = 3
warmup_steps = 0
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Gradient Accumulation & Logging
gradient_accumulation_steps = 2
writer = SummaryWriter(log_dir="training_logs")

# Training Loop
for epoch in range(num_epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    loss_total = 0
    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Gradient Accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + step)
            loss_total += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    # Save model checkpoint after every epoch
    model.save_pretrained(f"models/checkpoint_epoch_{epoch + 1}")
    writer.add_scalar('Average Loss', loss_total / len(train_dataloader), epoch)

writer.close()
print("Training complete!")
