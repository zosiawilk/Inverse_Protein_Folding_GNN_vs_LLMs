import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from tqdm import tqdm
import numpy as np
import argparse

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
        return 0.0, 0, 0
    
    # Calculate recovery for the overlapping part
    min_len = min(len(pred_aa), len(target_aa))
    matches = sum(1 for i in range(min_len) if pred_aa[i] == target_aa[i])
    
    # Recovery rate based on target length (standard metric)
    recovery = matches / len(target_aa)
    
    return recovery, matches, len(target_aa)

def calculate_perplexity(model, dataloader, device='cuda'):
    """Calculate perplexity on the dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            # Weight by number of tokens
            mask = batch['labels'] != -100
            n_tokens = mask.sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss

def evaluate_model(model_path, data_path, output_file, device='cuda', 
                   max_samples=None, dataset_name='test'):
    """
    Evaluate the model and calculate sequence recovery rate.
    """
    print("="*60)
    print(f"EVALUATING MODEL ON {dataset_name.upper()} SET")
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
    total_matches = 0
    total_positions = 0
    
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
        recovery, matches, total = calculate_sequence_recovery(pred_sequence, target_sequence)
        recoveries.append(recovery)
        total_matches += matches
        total_positions += total
        
        # Store results
        result = {
            'name': sample['name'],
            'target': target_sequence,
            'prediction': pred_sequence,
            'recovery': recovery,
            'matches': matches,
            'target_length': len(target_sequence.split()),
            'pred_length': len(pred_sequence.split())
        }
        results.append(result)
        
        # Print first few examples
        if i < 3:
            print(f"\n--- Example {i+1} ({sample['name']}) ---")
            print(f"Target:     {target_sequence[:80]}...")
            print(f"Prediction: {pred_sequence[:80]}...")
            print(f"Recovery: {recovery:.2%} ({matches}/{total})")
    
    # Calculate statistics
    mean_recovery = np.mean(recoveries)
    std_recovery = np.std(recoveries)
    median_recovery = np.median(recoveries)
    global_recovery = total_matches / total_positions if total_positions > 0 else 0
    
    # Save detailed results
    output_data = {
        'dataset': dataset_name,
        'model_path': model_path,
        'overall_statistics': {
            'mean_recovery': float(mean_recovery),
            'std_recovery': float(std_recovery),
            'median_recovery': float(median_recovery),
            'global_recovery': float(global_recovery),
            'total_matches': int(total_matches),
            'total_positions': int(total_positions),
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
    print(f"Dataset: {dataset_name}")
    print(f"Number of samples: {len(results)}")
    print(f"Mean sequence recovery: {mean_recovery:.4f} ({mean_recovery*100:.2f}%)")
    print(f"Std sequence recovery: {std_recovery:.4f} ({std_recovery*100:.2f}%)")
    print(f"Median sequence recovery: {median_recovery:.4f} ({median_recovery*100:.2f}%)")
    print(f"Global recovery: {global_recovery:.4f} ({global_recovery*100:.2f}%)")
    print(f"  (Total matches: {total_matches} / {total_positions})")
    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)
    
    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate protein inverse folding model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test/valid data')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to save results JSON')
    parser.add_argument('--dataset_name', type=str, default='test',
                       help='Name of dataset (test/valid)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (for quick testing)')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        args.model_path, 
        args.data_path, 
        args.output_file,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples
    )
