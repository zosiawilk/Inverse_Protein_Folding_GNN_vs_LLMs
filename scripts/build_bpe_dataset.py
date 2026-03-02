import json
import os
from collections import Counter

# ── BPE implementation ───────────────────────────────────────────────────────

def get_pairs(sequences):
    """Count all adjacent token pairs across all sequences."""
    pairs = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pairs[(seq[i], seq[i+1])] += 1
    return pairs

def merge_pair(sequences, pair):
    """Merge all occurrences of pair in all sequences."""
    merged = ''.join(pair)
    new_sequences = []
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_sequences.append(new_seq)
    return new_sequences

def learn_bpe(sequences, num_merges=500):
    """
    Learn BPE merges from sequences.
    sequences: list of lists of tokens (e.g. [['a','d','k'], ['v','v','l']])
    Returns: list of merge rules (pair -> merged_token)
    """
    print(f"Learning BPE with {num_merges} merges...")
    merges = []
    
    for i in range(num_merges):
        pairs = get_pairs(sequences)
        if not pairs:
            break
        
        # find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        sequences = merge_pair(sequences, best_pair)
        
        if i % 50 == 0:
            vocab_size = len(set(t for seq in sequences for t in seq))
            print(f'  Merge {i}/{num_merges} | pair: {best_pair} | vocab size: {vocab_size}')
    
    return merges, sequences

def apply_bpe(sequence, merges):
    """Apply learned BPE merges to a new sequence."""
    tokens = list(sequence)  # start with individual characters
    for pair in merges:
        tokens = merge_pair([tokens], pair)[0]
    return tokens

# ── load data ────────────────────────────────────────────────────────────────

# load 3Di tokens (we run BPE on top of 3Di)
def load_fasta(path):
    sequences = {}
    with open(path) as f:
        name = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                raw = line[1:].split()[0]
                name = raw.replace('_', '.')
            else:
                if name:
                    sequences[name] = line
    return sequences

print("Loading 3Di tokens...")
tokens_3di = load_fasta(
    '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath_3di/cath_3di.fasta'
)

# load splits
with open('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath/chain_set_splits.json') as f:
    splits = json.load(f)

# load AA sequences
aa_seqs = {}
with open('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath/chain_set.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        aa_seqs[entry['name']] = entry['seq']

# ── learn BPE on training set only ───────────────────────────────────────────
train_names = splits['train']
train_sequences = []
train_valid_names = []

for name in train_names:
    if name in tokens_3di:
        train_sequences.append(list(tokens_3di[name]))
        train_valid_names.append(name)

print(f"Learning BPE on {len(train_sequences)} training proteins...")
merges, _ = learn_bpe(train_sequences, num_merges=500)

# save merges for later use
merges_path = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/bpe_merges.json'
with open(merges_path, 'w') as f:
    json.dump([list(m) for m in merges], f)
print(f"Saved {len(merges)} BPE merges to {merges_path}")

# ── build dataset ─────────────────────────────────────────────────────────────
out_dir = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/llm_dataset_bpe'
os.makedirs(out_dir, exist_ok=True)

for split_name, names in [('train',  splits['train']),
                           ('valid',  splits['validation']),
                           ('test',   splits['test'])]:
    examples = []
    skipped  = 0
    for name in names:
        if name not in tokens_3di or name not in aa_seqs:
            skipped += 1
            continue

        # apply BPE to get compressed token sequence
        bpe_tokens = apply_bpe(tokens_3di[name], merges)
        aa_sequence = ' '.join(list(aa_seqs[name]))

        # format: tokens joined by spaces
        struct_tokens = ' '.join(bpe_tokens)

        examples.append({
            'name':            name,
            'input':           f'Structure: {struct_tokens}',
            'output':          f'Sequence: {aa_sequence}',
            'original_length': len(tokens_3di[name]),
            'bpe_length':      len(bpe_tokens),
        })

    out_path = os.path.join(out_dir, f'{split_name}.json')
    with open(out_path, 'w') as f:
        json.dump(examples, f, indent=2)

    if examples:
        avg_orig = sum(e['original_length'] for e in examples) / len(examples)
        avg_bpe  = sum(e['bpe_length']      for e in examples) / len(examples)
        print(f'{split_name}: {len(examples)} examples | '
              f'avg length {avg_orig:.0f} → {avg_bpe:.0f} tokens ({100*avg_bpe/avg_orig:.0f}% of original)')
    print(f'  skipped: {skipped}')

print(f"\nDataset saved to {out_dir}")
print("Example:")
print(json.dumps(examples[0], indent=2))

