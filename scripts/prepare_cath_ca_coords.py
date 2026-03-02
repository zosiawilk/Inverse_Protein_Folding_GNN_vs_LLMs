import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_ca_coords_from_cath(item, decimal_places=1):
    """
    Extract CA coordinates from CATH data format.
    """
    ca_coords = np.array(item['coords']['CA'])
    
    # Remove NaN values
    valid_mask = ~np.isnan(ca_coords).any(axis=1)
    ca_coords = ca_coords[valid_mask]
    
    # Get corresponding sequence (skip NaN positions)
    full_sequence = list(item['seq'])
    sequence = [aa for i, aa in enumerate(full_sequence) if valid_mask[i]]
    
    # Flatten coordinates
    coords_flat = ca_coords.flatten()
    coords_rounded = np.round(coords_flat, decimal_places)
    
    coord_string = ' '.join(str(x) for x in coords_rounded)
    sequence_string = ' '.join(sequence)
    
    return coord_string, sequence_string, len(sequence)

def load_jsonl(filepath):
    """Load a JSONL file (one JSON object per line)."""
    data = {}
    print(f"Loading {filepath}...")
    with open(filepath) as f:
        for line in tqdm(f, desc="Reading JSONL"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Use 'name' as key
            data[item['name']] = item
    print(f"✓ Loaded {len(data)} proteins")
    return data

def process_cath_with_splits(chain_set_file, splits_file, output_dir, 
                             decimal_places=1, max_length=400):
    """
    Process CATH dataset with chain_set.jsonl and chain_set_splits.json format.
    """
    print("="*60)
    print("Processing CATH dataset")
    print("="*60)
    
    # Load chain set (JSONL format)
    chain_set = load_jsonl(chain_set_file)
    
    # Load splits (regular JSON)
    print(f"\nLoading splits...")
    with open(splits_file) as f:
        splits = json.load(f)
    
    for split_name, chain_ids in splits.items():
        print(f"  {split_name}: {len(chain_ids)} proteins")
    
    # Process each split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    for split_name, chain_ids in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        output_data = []
        skipped = 0
        too_long = 0
        not_found = 0
        
        for chain_id in tqdm(chain_ids, desc=f"Converting {split_name}"):
            try:
                if chain_id not in chain_set:
                    not_found += 1
                    continue
                
                item = chain_set[chain_id]
                
                coord_string, sequence, num_residues = extract_ca_coords_from_cath(
                    item, decimal_places
                )
                
                if max_length and num_residues > max_length:
                    too_long += 1
                    continue
                
                num_coord_tokens = len(coord_string.split())
                
                output_item = {
                    'name': chain_id,
                    'input': f"Coordinates: {coord_string}",
                    'output': f"Sequence: {sequence}",
                    'num_residues': num_residues,
                    'num_coord_tokens': num_coord_tokens,
                    'cath': item.get('CATH', [])
                }
                
                output_data.append(output_item)
                
            except Exception as e:
                print(f"\nError processing {chain_id}: {e}")
                skipped += 1
                continue
        
        # Save output
        output_file = output_dir / f'{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Statistics
        if len(output_data) > 0:
            coord_tokens = [item['num_coord_tokens'] for item in output_data]
            residues = [item['num_residues'] for item in output_data]
            
            print(f"\n{split_name} Statistics:")
            print(f"  Saved: {len(output_data)} samples")
            if not_found > 0:
                print(f"  Not found: {not_found}")
            if skipped > 0:
                print(f"  Skipped (errors): {skipped}")
            if too_long > 0:
                print(f"  Filtered (too long): {too_long}")
            print(f"  Residues - Mean: {np.mean(residues):.1f}, Max: {max(residues)}, Min: {min(residues)}")
            print(f"  Coord tokens - Mean: {np.mean(coord_tokens):.1f}, Max: {max(coord_tokens)}, Min: {min(coord_tokens)}")
            
            all_stats[split_name] = {
                'num_samples': len(output_data),
                'mean_residues': float(np.mean(residues)),
                'mean_tokens': float(np.mean(coord_tokens))
            }
        else:
            print(f"\n⚠️  No samples for {split_name}!")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split_name, stats in all_stats.items():
        print(f"{split_name:10s}: {stats['num_samples']:5d} samples, "
              f"{stats['mean_residues']:5.1f} residues, "
              f"{stats['mean_tokens']:6.1f} tokens")
    print(f"\n✅ All splits saved to: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CATH data to CA coordinate format')
    parser.add_argument('--cath_dir', type=str, required=True,
                       help='Directory containing chain_set.jsonl and chain_set_splits.json')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed files')
    parser.add_argument('--decimal_places', type=int, default=1,
                       help='Decimal places for coordinate rounding (default: 1)')
    parser.add_argument('--max_length', type=int, default=400,
                       help='Maximum protein length in residues (default: 400)')
    
    args = parser.parse_args()
    
    cath_dir = Path(args.cath_dir)
    
    # Look for JSONL file
    chain_set_file = cath_dir / 'chain_set.jsonl'
    splits_file = cath_dir / 'chain_set_splits.json'
    
    if not chain_set_file.exists():
        print(f"Error: {chain_set_file} not found!")
        exit(1)
    if not splits_file.exists():
        print(f"Error: {splits_file} not found!")
        exit(1)
    
    process_cath_with_splits(
        chain_set_file,
        splits_file,
        args.output_dir,
        decimal_places=args.decimal_places,
        max_length=args.max_length
    )