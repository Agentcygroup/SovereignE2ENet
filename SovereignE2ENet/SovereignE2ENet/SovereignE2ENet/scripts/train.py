import torch
from trafrom trafrom trafromoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_datafrom datasets import lata import DataLoader
from tqdm import tqdm

# Check device compatibility
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load the custom dataset
dataset = load_dataset('json', data_files='data/instructions.json', split='train')
def tokenize_function(examples):
    return tokenizer(examples['instruction'], truncation=True, padding='max_length', max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets.shuffle(seed=42)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    # Save model checkpoint after every epoch
    model.save_pretrained(f"models/checkpoint_epoch_{epoch + 1}")

print("Training complete!")
