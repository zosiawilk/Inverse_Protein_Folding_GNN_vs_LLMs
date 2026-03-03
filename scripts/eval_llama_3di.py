import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MODEL_PATH = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/best_model'
DATA_PATH  = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset/test.json'
OUTPUT_FILE = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/eval_results.json'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B',
    quantization_config=bnb_config,
    device_map='auto',
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()
print("Model loaded.")

print("Loading test data...")
with open(DATA_PATH) as f:
    test_data = json.load(f)
test_data = test_data[:100] # Limit to 100 samples for quick evaluation
print(f"Test samples: {len(test_data)}")

recoveries = []
results = []

for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
    prompt = item['input'] + '\nSequence:'
    inputs = tokenizer(prompt, return_tensors='pt', 
                      truncation=True, max_length=3800).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if 'Sequence:' in generated:
        pred_seq = generated.split('Sequence:')[-1].strip().split()
    else:
        pred_seq = []

    true_seq = item['output'].replace('Sequence:', '').strip().split()

    if len(pred_seq) == 0 or len(true_seq) == 0:
        recoveries.append(0.0)
        continue

    matches = sum(p == t for p, t in zip(pred_seq, true_seq))
    recovery = matches / len(true_seq)
    recoveries.append(recovery)

    results.append({
        'name': item.get('name', str(i)),
        'recovery': recovery,
        'true_len': len(true_seq),
        'pred_len': len(pred_seq)
    })

    if i < 3:
        print(f"\n--- Example {i+1} ---")
        print(f"True: {' '.join(true_seq[:20])}")
        print(f"Pred: {' '.join(pred_seq[:20])}")
        print(f"Recovery: {recovery:.3f}")

print("\n=== RESULTS ===")
print(f"Proteins evaluated: {len(recoveries)}")
print(f"Mean recovery:   {np.mean(recoveries):.4f}")
print(f"Median recovery: {np.median(recoveries):.4f}")
print(f"Std:             {np.std(recoveries):.4f}")

with open(OUTPUT_FILE, 'w') as f:
    json.dump({
        'mean_recovery': float(np.mean(recoveries)),
        'median_recovery': float(np.median(recoveries)),
        'std_recovery': float(np.std(recoveries)),
        'results': results
    }, f, indent=2)

print(f"Saved to {OUTPUT_FILE}")
