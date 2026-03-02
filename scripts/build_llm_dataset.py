import json
import os

def load_fasta(path):
    sequences = {}
    with open(path) as f:
        name = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # foldseek gives "12as_A DESCRIPTION..."
                # take just the first part before space
                raw = line[1:].split()[0]
                # normalise to match splits format: "12as_A" -> "12as.A"
                name = raw.replace('_', '.')
            else:
                if name:
                    sequences[name] = line
    return sequences

def load_aa_sequences(path):
    sequences = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            sequences[entry['name']] = entry['seq']
    return sequences

# load splits
with open('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath/chain_set_splits.json') as f:
    splits = json.load(f)

# load data
print("Loading 3Di tokens...")
tokens_3di = load_fasta('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath_3di/cath_3di.fasta')
print(f"  loaded {len(tokens_3di)} 3Di sequences")
print(f"  example keys: {list(tokens_3di.keys())[:3]}")

print("Loading AA sequences...")
aa_seqs = load_aa_sequences('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath/chain_set.jsonl')
print(f"  loaded {len(aa_seqs)} AA sequences")
print(f"  example keys: {list(aa_seqs.keys())[:3]}")

out_dir = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset'
os.makedirs(out_dir, exist_ok=True)

for split_name, names in [('train',  splits['train']),
                           ('valid',  splits['validation']),
                           ('test',   splits['test'])]:
    examples = []
    skipped  = 0
    for name in names:
        if name not in tokens_3di:
            skipped += 1
            continue
        if name not in aa_seqs:
            skipped += 1
            continue

        struct_tokens = ' '.join(list(tokens_3di[name]))
        aa_sequence   = ' '.join(list(aa_seqs[name]))

        examples.append({
            'name':   name,
            'input':  f'Structure: {struct_tokens}',
            'output': f'Sequence: {aa_sequence}'
        })

    out_path = os.path.join(out_dir, f'{split_name}.json')
    with open(out_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f'{split_name}: {len(examples)} examples saved, {skipped} skipped')

print(f"\nDataset saved to {out_dir}")
if examples:
    print("Example entry:")
    print(json.dumps(examples[0], indent=2))
