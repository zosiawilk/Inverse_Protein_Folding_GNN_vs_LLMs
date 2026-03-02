import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from tqdm import tqdm
import numpy as np

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
        
        # Just tokenize the input for generation
        input_ids = self.tokenizer(input_text, add_special_tokens=True, 
                                   return_tensors='pt')['input_ids']
        
        # Keep the target sequence for comparison
        target_sequence = output_text
        
        return {
            'input_ids': input_ids.squeeze(0),
            'target_sequence': target_sequence,
            'name': item['name']
        }

def calculate_sequence_recovery(pred_seq, target_seq):
    """
    Calculate amino acid level recovery rate.
    Both sequences should be space-separated strings of amino acids.
    """
    pred_aa = pred_seq.split()
    target_aa = target_seq.split()
    
    if len(pred_aa) == 0 or len(target_aa) == 0:
        return 0.0
    
    # Calculate recovery for the overlapping part
    min_len = min(len(pred_aa), len(target_aa))
    matches = sum(1 for i in range(min_len) if pred_aa[i] == target_aa[i])
    
    # Recovery rate based on target length (standard metric)
    recovery = matches / len(target_aa)
    
    return recovery

def evaluate_model(model_path, data_path, output_file, device='cuda', max_samples=None):
    """
    Evaluate the model and calculate sequence recovery rate.
    """
    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {device}")
    print()
    
    # Load tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✓ Tokenizer loaded")
    
    # Load model
    print("\n[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model.eval()
    print(f"✓ Model loaded")
    
    # Load dataset
    print("\n[3/4] Loading dataset...")
    dataset = ProteinDataset(data_path, tokenizer)
    if max_samples:
        dataset.data = dataset.data[:max_samples]
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Evaluate
    print("\n[4/4] Generating predictions...")
    results = []
    recoveries = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        
        # Generate sequence
        with torch.no_grad():
            generated = model.generate(
                input_ids=sample['input_ids'].unsqueeze(0).to(device),
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                temperature=1.0
            )
        
        # Decode prediction
        full_output = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract only the generated sequence part (after "Sequence:")
        if 'Sequence:' in full_output:
            pred_sequence = full_output.split('Sequence:')[-1].strip()
        else:
            pred_sequence = full_output.strip()
        
        target_sequence = sample['target_sequence']
        
        # Calculate recovery
        recovery = calculate_sequence_recovery(pred_sequence, target_sequence)
        recoveries.append(recovery)
        
        # Store results
        result = {
            'name': sample['name'],
            'target': target_sequence,
            'prediction': pred_sequence,
            'recovery': recovery,
            'target_length': len(target_sequence.split()),
            'pred_length': len(pred_sequence.split())
        }
        results.append(result)
        
        # Print first few examples
        if i < 5:
            print(f"\n--- Example {i+1} ---")
            print(f"Name: {sample['name']}")
            print(f"Target:     {target_sequence[:100]}...")
            print(f"Prediction: {pred_sequence[:100]}...")
            print(f"Recovery: {recovery:.2%}")
    
    # Calculate statistics
    mean_recovery = np.mean(recoveries)
    std_recovery = np.std(recoveries)
    median_recovery = np.median(recoveries)
    
    # Save detailed results
    output_data = {
        'overall_statistics': {
            'mean_recovery': mean_recovery,
            'std_recovery': std_recovery,
            'median_recovery': median_recovery,
            'num_samples': len(results)
        },
        'per_sample_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {len(results)}")
    print(f"Mean sequence recovery: {mean_recovery:.2%} ± {std_recovery:.2%}")
    print(f"Median sequence recovery: {median_recovery:.2%}")
    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)
    
    return output_data

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/best_model'
    DATA_PATH = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset/valid.json'
    OUTPUT_FILE = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/evaluation_results.json'
    
    # You can also evaluate on test set if you have one
    # DATA_PATH = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset/test.json'
    
    # Run evaluation
    # Set max_samples=50 for quick test, or None for full evaluation
    results = evaluate_model(MODEL_PATH, DATA_PATH, OUTPUT_FILE, max_samples=None)
