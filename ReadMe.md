# Assessing the Transferability of Text LLMs for Protein Inverse Folding via Structural Tokenization

**Zofia Wilk** | University of Cambridge | Geometric Deep Learning (L65) Mini-Project

---

## Overview

Protein inverse folding is the task of designing amino acid sequences that fold into a given 3D backbone structure. This project investigates whether **fine-tuned text-pretrained LLMs** can compete with **geometric Graph Neural Networks (GNNs)** on this task, and which structural tokenization strategy yields the best results.

We fine-tune **LLaMA-3-8B** on the CATH 4.2 benchmark using three tokenization schemes and compare against GNN baselines (ProteinMPNN and PiFold). We also probe how much of PiFold's performance stems from its hand-crafted geometric features by training a variant on raw backbone coordinates.

---

## Results Summary

| Model | Type | Input | Recovery (%) |
|---|---|---|---|
| ProteinMPNN | GNN | Hand-crafted features | 45.96† |
| PiFold | GNN | Hand-crafted features | 51.66† |
| PiFold (reproduced) | GNN | Hand-crafted features | 51.35 |
| PiFold (raw coords) | GNN | Raw N/Cα/C/O | 50.44 |
| LLaMA-3-8B + 3Di | LLM | Foldseek 3Di tokens | 8.64 |
| LLaMA-3-8B + Cα coords | LLM | Raw Cα coordinates | 5.44 |

† Results taken directly from original papers.

**Key findings:**
- GNNs significantly outperform fine-tuned LLMs on this task under comparable compute budgets.
- PiFold trained on raw coordinates (50.44%) nearly matches its hand-crafted-feature variant (51.35%), suggesting the GNN architecture itself — not feature engineering — drives most of the performance.
- LLM models collapse into repetitive outputs, indicating insufficient training and/or a large domain gap from natural language pretraining.

---

## Repository Structure

```
.
├── data/                   # Dataset preparation and preprocessing scripts
├── models/
│   ├── pifold/             # PiFold GNN (original + raw coords variant)
│   └── llm/                # LLaMA-3-8B fine-tuning code (3Di, BPE, Cα)
├── tokenization/
│   ├── foldseek_3di.py     # Foldseek 3Di token generation
│   ├── geometric_bpe.py    # BPE over 3Di sequences
│   └── raw_coords.py       # Raw Cα coordinate serialization
├── evaluation/             # Sequence recovery metrics and prediction examples
└── README.md
```

---

## Methods

### Dataset
All models are evaluated on the **CATH 4.2 benchmark**, using the standard splits from ProteinMPNN/PiFold:
- **Training:** 18,024 chains
- **Validation:** 608 chains
- **Test:** 1,120 chains
- Protein lengths: 40–500 residues (median 217)

### Structural Tokenization

Three representations of protein 3D structure are explored:

**1. Raw Cα Coordinates**
Each residue's alpha-carbon position is serialized as `x y z` rounded to one decimal place, producing ~3 tokens per residue.
```
Input:  Coordinates: 4.8 -7.3 5.9 7.0 -6.7 9.0 ...
Output: Sequence: E S R L D R ...
```

**2. Foldseek 3Di Tokens**
A 20-letter structural alphabet derived from tertiary interactions, one token per residue.
```
Input:  Structure: D V V V V V L V V V L V V L ...
Output: Sequence: M Y V A S W Q D Y H S D F S ...
```

**3. Geometric BPE**
BPE applied to 3Di sequences with 500 merge rules, compressing average proteins from 231 → 102 tokens (44% compression).
```
Input:  Structure: DP FF DF Q DWDW LV GH DT CP ...
Output: Sequence: S N A K V T V G K S A P Y F ...
```

### LLM Fine-tuning

- **Base model:** LLaMA-3-8B
- **Adaptation:** LoRA (Low-Rank Adaptation), ~6.8M trainable parameters
- **Quantization:** 8-bit for 3Di/BPE; 4-bit for raw coordinates
- **Training:** Adam optimizer, 10 epochs (3 epochs in practice due to compute constraints), gradient clipping at 1.0
- **Context lengths:** 1,024 tokens (3Di/BPE), 4,096 tokens (Cα)

### GNN Baselines

- **ProteinMPNN:** Results taken from the original paper (47.9% / 45.96% median recovery).
- **PiFold (reproduced):** 10-layer graph transformer, hidden dim 128, trained for 100 epochs with Adam + OneCycleLR.
- **PiFold (raw coords):** All hand-crafted features replaced with raw backbone coordinates in a local frame.

---

## Setup & Installation

```bash
git clone https://github.com/zosiawilk/Inverse_Protein_Folding_GNN_vs_LLMs.git
cd Inverse_Protein_Folding_GNN_vs_LLMs
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- Hugging Face `transformers` and `peft` (for LoRA)
- `bitsandbytes` (for quantization)
- [Foldseek](https://github.com/steineggerlab/foldseek) (for 3Di tokenization)

---

## Running Experiments

### GNN Training
```bash
# PiFold with hand-crafted features
python models/pifold/train.py --features geometric

# PiFold with raw coordinates
python models/pifold/train.py --features raw_coords
```

### LLM Fine-tuning
```bash
# Fine-tune with 3Di tokens
python models/llm/finetune.py --tokenization 3di

# Fine-tune with Geometric BPE
python models/llm/finetune.py --tokenization bpe

# Fine-tune with raw Cα coordinates
python models/llm/finetune.py --tokenization coords --max_length 4096 --quantization 4bit
```

### Evaluation
```bash
python evaluation/evaluate.py --model [pifold|llama_3di|llama_bpe|llama_coords]
```

---

## Computational Notes

All experiments were run on the **Cambridge HPC cluster**, consuming ~63 GPU hours in total. Key constraints:

- LLaMA-3-8B requires substantial GPU memory; 8-bit quantization was used throughout, with 4-bit for the coordinate model.
- Gradient checkpointing was enabled to reduce activation memory.
- Due to compute budget limits, LLM models were trained for 3 epochs on a subset of the training data (10–20%), which is likely insufficient for convergence.
- A fully converged LLM run is estimated to require ~100 GPU hours per model.

---

## Future Work

- Train for more epochs on the full CATH training set with proper compute allocation.
- Explore coordinate normalization (centroid-centering, radius-of-gyration scaling) to stabilize Cα model training.
- Implement the full Geometric BPE method from [Guo et al., 2024](https://arxiv.org/abs/2511.11758), including differentiable inverse kinematics at merge junctions.
- Evaluate on additional benchmarks: TS50 and TS500.
- Compare across different LLM architectures (GPT variants, LLaMA-2).

---

## Citation

If you use this work, please cite:

```
@misc{wilk2024inversefolding,
  author    = {Zofia Wilk},
  title     = {Assessing the Transferability of Text LLMs for Protein Inverse Folding via Structural Tokenization},
  year      = {2024},
  note      = {L65 Mini-Project, University of Cambridge}
}
```

---

## Acknowledgements

Thank you to my supervisor Vladimir for proposing the project and providing guidance throughout.
