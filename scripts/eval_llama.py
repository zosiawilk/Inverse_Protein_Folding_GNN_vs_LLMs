import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── config ─────────────────────────────────────────────────────────────────
BASE_MODEL  = 'meta-llama/Meta-Llama-3-8B'
LORA_MODEL  = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/best_model'
TEST_DATA   = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset/test.json'

# ── load model ─────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map='auto'
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()
print("Model loaded.")

# ── load test data ──────────────────────────────────────────────────────────
with open(TEST_DATA) as f:
    test_data = json.load(f)

print(f"Evaluating on {len(test_data)} test proteins...")

# ── evaluate ────────────────────────────────────────────────────────────────
recovery_scores = []

for i, item in enumerate(test_data):
    prompt = item['input'] + '\nSequence:'
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if 'Sequence:' in generated:
        pred_seq = generated.split('Sequence:')[-1].strip().replace(' ', '')
    else:
        pred_seq = ''

    true_seq = item['output'].replace('Sequence:', '').strip().replace(' ', '')

    min_len = min(len(pred_seq), len(true_seq))
    if min_len == 0:
        continue

    matches = sum(p == t for p, t in zip(pred_seq[:min_len], true_seq[:min_len]))
    recovery = matches / len(true_seq)
    recovery_scores.append(recovery)

    if i % 50 == 0:
        print(f'  {i}/{len(test_data)} | Recovery so far: {np.mean(recovery_scores):.4f}')
        print(f'  True: {true_seq[:30]}')
        print(f'  Pred: {pred_seq[:30]}')

# ── results ─────────────────────────────────────────────────────────────────
print(f'\n=== Llama + 3Di Results ===')
print(f'Proteins evaluated: {len(recovery_scores)}')
print(f'Mean recovery:   {np.mean(recovery_scores):.4f}')
print(f'Median recovery: {np.median(recovery_scores):.4f}')
print(f'Std:             {np.std(recovery_scores):.4f}')
print(f'Min:             {np.min(recovery_scores):.4f}')
print(f'Max:             {np.max(recovery_scores):.4f}')
