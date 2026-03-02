import json
import os
import urllib.request
import time

# load all protein names from splits
with open('/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath/chain_set_splits.json') as f:
    splits = json.load(f)

all_names = splits['train'] + splits['validation'] + splits['test']

# extract unique PDB IDs (first 4 chars, e.g. "1ABC" from "1ABC_A")
pdb_ids = list(set([name[:4] for name in all_names]))
print(f"Downloading {len(pdb_ids)} PDB files...")

out_dir = '/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath_pdb'
os.makedirs(out_dir, exist_ok=True)

failed = []
for i, pdb_id in enumerate(pdb_ids):
    out_path = os.path.join(out_dir, f'{pdb_id}.pdb')
    
    # skip if already downloaded
    if os.path.exists(out_path):
        continue
    
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    try:
        urllib.request.urlretrieve(url, out_path)
        if i % 100 == 0:
            print(f'{i}/{len(pdb_ids)} done')
        time.sleep(0.1)  # be polite to the server
    except Exception as e:
        print(f'Failed: {pdb_id} — {e}')
        failed.append(pdb_id)

print(f"Done. Failed: {len(failed)}")
print(failed)
