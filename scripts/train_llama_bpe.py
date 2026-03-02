import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import sys

# --- CONFIGURATION ---
DATA_DIR = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset_bpe'
OUT_DIR  = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_bpe'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'

# Efficiency settings
MAX_LENGTH = 1024
BATCH_SIZE = 4       # Increase to 8 if you have 80GB GPU
GRAD_ACCUM = 4       # Effective batch size = 4 * 4 = 16
LR = 2e-4
EPOCHS = 3           # 3 epochs is usually enough for LoRA fine-tuning

class ProteinDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        print(f"Loading {path}...")
        with open(path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Format Inputs
        input_text  = item['input'] + '\nSequence:'
        output_text = item['output'].replace('Sequence:', '').strip()
        
        # 2. Tokenize separately to handle prompt vs response masking
        # add_special_tokens=True adds the BOS token to the start
        input_ids  = self.tokenizer(input_text,  add_special_tokens=True)['input_ids']
        output_ids = self.tokenizer(output_text, add_special_tokens=False)['input_ids']
        
        # 3. Add EOS token to the end so model knows when to stop
        output_ids = output_ids + [self.tokenizer.eos_token_id]
        
        # 4. Concatenate and Truncate
        full_ids = input_ids + output_ids
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            
        # 5. Calculate lengths for masking
        input_len = len(input_ids)
        if input_len > self.max_length:
            input_len = self.max_length
            
        # 6. Padding
        pad_len = self.max_length - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        
        # 7. Create Tensors
        input_ids_t = torch.tensor(full_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        labels_t = input_ids_t.clone()
        
        # 8. Masking (Ignore prompt and padding in loss calculation)
        # -100 is the standard PyTorch ignore index
        labels_t[:input_len] = -100 
        labels_t[input_ids_t == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids':      input_ids_t,
            'attention_mask': attention_mask_t,
            'labels':         labels_t
        }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    print("Loading model...")
    # Use bfloat16 for Ampere GPUs (more efficient/stable than float16)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using data type: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map='auto',
        use_cache=False # Disable KV cache for training to save VRAM
    )

    # Enable Gradient Checkpointing (Saves massive VRAM)
    model.gradient_checkpointing_enable()

    # LoRA Config - Targeting all linear layers helps learn new syntax better
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = ProteinDataset(f'{DATA_DIR}/train.json', tokenizer, MAX_LENGTH)
    valid_dataset = ProteinDataset(f'{DATA_DIR}/valid.json', tokenizer, MAX_LENGTH)
    
    # Num workers loads data in background
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_valid_loss = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                labels=batch['labels'].cuda()
            )
            
            # Normalize loss by accumulation steps
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Record actual loss
            train_loss += loss.item() * GRAD_ACCUM
            
            if i % 50 == 0:
                print(f'  Epoch {epoch+1} step {i}/{len(train_loader)} loss: {loss.item()*GRAD_ACCUM:.4f}')
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
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
        
        avg_valid_loss = valid_loss / len(valid_loader)

        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}')

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_path = f'{OUT_DIR}/best_model'
            print(f'  -> Saving best model to {save_path}')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("Training done.")

