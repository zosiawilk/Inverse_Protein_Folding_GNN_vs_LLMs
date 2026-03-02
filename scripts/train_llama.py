import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  
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
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
#Add quantization config to load the model in 8-bit and further reduce memory usage with 4-bit compute dtype for faster training with LoRA
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

#Use quantization and low_cpu_mem_usage to reduce memory usage when loading the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    low_cpu_mem_usage=True
)

# Prepare for k-bit training to further reduce memory usage and enable training with LoRA 
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj'],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading datasets...")
train_dataset = ProteinDataset(f'{DATA_DIR}/train.json', tokenizer)
valid_dataset = ProteinDataset(f'{DATA_DIR}/valid.json', tokenizer)
print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=4) #chnage number of workers to 0 because of too big memory usage 
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)  # change lr to 3e-3 for faster convergence with LoRA because it refused to learn 
best_valid_loss = float('inf')


for epoch in range(10):
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
        train_loss += loss.item()
        if i % 100 == 0:
            print(f'  Epoch {epoch+1} step {i}/{len(train_loader)} loss: {loss.item():.4f}')
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

    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model.save_pretrained(f'{OUT_DIR}/best_model')
        tokenizer.save_pretrained(f'{OUT_DIR}/best_model')
        print(f'  -> saved best model (valid loss: {valid_loss:.4f})')

print("Training done.")
