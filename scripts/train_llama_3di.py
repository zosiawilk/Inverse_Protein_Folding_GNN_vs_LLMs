import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os

class ProteinDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        with open(path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text  = item['input'] + '\nSequence:'
        output_text = item['output'].replace('Sequence:', '').strip()
        input_ids  = self.tokenizer(input_text,  add_special_tokens=True)['input_ids']
        output_ids = self.tokenizer(output_text, add_special_tokens=False)['input_ids']
        output_ids = output_ids + [self.tokenizer.eos_token_id]
        full_ids = input_ids + output_ids
        full_ids = full_ids[:self.max_length]
        input_len = min(len(input_ids), self.max_length)
        pad_len = self.max_length - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        input_ids      = torch.tensor(full_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = input_ids.clone()
        labels[:input_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels
        }

DATA_DIR = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset'
OUT_DIR  = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'

# ← KEY SETTINGS FOR SPEED
NUM_EPOCHS = 3          # Reduced from 10
DATA_FRACTION = 0.2     # Use 20% of data
LEARNING_RATE = 5e-5    # Lower and more stable
BATCH_SIZE = 4

os.makedirs(OUT_DIR, exist_ok=True)

print("="*60)
print("TRAINING ON 3Di (FAST MODE)")
print("="*60)
print(f"Epochs: {NUM_EPOCHS}")
print(f"Data fraction: {DATA_FRACTION*100}%")
print(f"Learning rate: {LEARNING_RATE}")

print("\n[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("\n[2/6] Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    low_cpu_mem_usage=True
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n[3/6] Loading datasets...")
train_dataset = ProteinDataset(f'{DATA_DIR}/train.json', tokenizer)
valid_dataset = ProteinDataset(f'{DATA_DIR}/valid.json', tokenizer)

# ← Reduce training data for speed
random.seed(42)
random.shuffle(train_dataset.data)
train_dataset.data = train_dataset.data[:int(len(train_dataset.data) * DATA_FRACTION)]
print(f"✓ Train (reduced): {len(train_dataset)} samples")
print(f"✓ Valid: {len(valid_dataset)} samples")

print("\n[4/6] Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print("\n[5/6] Setting up optimizer and scheduler...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler with warmup
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
print(f"✓ Total steps: {total_steps}")
print(f"✓ Warmup steps: {warmup_steps}")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].cuda(),
            attention_mask=batch['attention_mask'].cuda(),
            labels=batch['labels'].cuda()
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # ← Update LR
        train_loss += loss.item()
        
        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'  Step {i}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}')
    
    train_loss /= len(train_loader)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                labels=batch['labels'].cuda()
            )
            valid_loss += outputs.loss.item()
    valid_loss /= len(valid_loader)

    print(f"\n  RESULTS Epoch {epoch+1}:")
    print(f"    Train Loss: {train_loss:.4f}")
    print(f"    Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f"  ✓ New best model!")
        model.save_pretrained(f'{OUT_DIR}/best_model')
        tokenizer.save_pretrained(f'{OUT_DIR}/best_model')

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"Best validation loss: {best_valid_loss:.4f}")
print("="*60)