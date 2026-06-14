# Project 3: New Topics in Machine Learning Research
## Topic: Foundations of Tensor Computations for AI (NeurIPS 2025)
---

## 1. Project Overview
This project focuses on research, discussion, and experiments based on the NeurIPS 2025 Tutorial **"Foundations of Tensor Computations for AI"**. The implementation is divided into three main components:

1. **Deep Learning - LoRA Rank Sweep**: Compares text classification performance (accuracy, loss) and the number of trainable parameters between two baselines (*frozen encoder*, *full fine-tuning*) and *Low-Rank Adaptation (LoRA)* under varying rank sweeps $r \in \{2, 4, 8\}$. A tiny Transformer model (`prajjwal1/bert-tiny`) is used on the GLUE SST-2 dataset.
2. **Probabilistic Circuits (PC) - CPJ Circuit**: Implements a binary probabilistic circuit (`BinaryCPJCircuit`) from scratch using PyTorch (viewed as a non-negative tensor factorization model ensuring smooth and decomposable properties). The model is trained via maximum likelihood (using Adam) on:
   - A toy 4-bit correlated distribution to verify the normalization property of the probability tensor (where the sum of probabilities across all 16 binary states equals 1.0).
   - A density estimation task on the Binarized MNIST dataset (784 variables).
3. **Data Mining - Tensor Decomposition**: Implements core tensor decompositions and advanced models from scratch using NumPy:
   - **CP-ALS**: Alternating Least Squares algorithm containing full manual implementations of mode-$n$ unfolding, Khatri-Rao product, and MTTKRP operations.
   - **CP-OPT**: Gradient-based CP tensor optimization using the L-BFGS-B optimizer.
   - **PARAFAC2**: Tensor factorization that handles irregular slice sizes.
   - **CMTF (Coupled Matrix-Tensor Factorization)**: Integrates dynamic EEG data (3rd-order tensor) and static metadata (matrix) by sharing a coupled Subject Factor.
   - **SVD/PCA vs. CP-ALS**: Compares the matrix unfolding approach against tensor decomposition on EEG signals, demonstrating that keeping the tensor structure yields much more interpretable spatio-temporal patterns (trial, spatial, and temporal factors are cleanly separated).

---

## 2. Directory Structure
```text
lab-3/
├── README.md                  # Project documentation (this file)
├── data-mining/               # Section 3: Data Mining & Tensor Decomposition
│   ├── src/                   # Core NumPy implementation (CP-ALS, CP-OPT, PARAFAC2, CMTF)
│   ├── notebooks/             # Jupyter Notebooks for EEG analysis & visualization
│   └── pyproject.toml         # Python environment configuration
├── deep-learning/             # Section 1: LoRA Rank Sweep
│   ├── configs/               # YAML configuration files for different training setups
│   ├── src/                   # PyTorch training, evaluation, and plotting modules
│   ├── notebooks/             # Jupyter Notebook tutorials for LoRA rank sweep
│   └── pyproject.toml         # Python environment configuration
├── probabilistic-circuit/     # Section 2: Probabilistic Circuits (CPJ/CP-T)
│   ├── src/                   # BinaryCPJCircuit implementation in PyTorch
│   ├── experiments/           # Training scripts on toy dataset and MNIST
│   ├── demo.ipynb             # End-to-end execution notebook (Kaggle-friendly)
│   └── pyproject.toml         # Python environment configuration
├── report/                    # Technical Report (LaTeX)
│   ├── content/               # Report chapters (.tex files)
│   ├── main.tex               # Main report LaTeX source file
│   └── build/                 # Generated PDF and build output
├── slide/                     # Presentation Slides (LaTeX Beamer)
│   ├── content/               # Slide chapters (.tex files)
│   ├── main.tex               # Main slides LaTeX source file
│   └── build/                 # Generated PDF and build output
└── scripts/                   # Utility scripts
    ├── compile_report.ps1     # PowerShell script to compile report PDF
    ├── compile_slide.ps1      # PowerShell script to compile slides PDF
    └── clear_artifacts.ps1    # PowerShell script to clean up build output & caches
```

---

## 3. Environment Setup
All subprojects use **`uv`** as their package manager. You can sync dependencies for each individual subproject by navigating to its directory:

```bash
# Example: Setting up the environment for deep-learning
cd deep-learning
uv sync
```

---

## 4. Running Experiments

### 4.1. Deep Learning: LoRA Rank Sweep
Run the following commands from the `deep-learning/` directory:
```bash
# Sync environment
uv sync

# Run training configurations
uv run python -m src.train --config configs/frozen.yaml
uv run python -m src.train --config configs/full_finetune.yaml
uv run python -m src.train --config configs/lora_r2.yaml
uv run python -m src.train --config configs/lora_r4.yaml
uv run python -m src.train --config configs/lora_r8.yaml

# Generate performance comparison plots
uv run python -m src.visualize --results results/runs --out results/plots
```
The output reports and figures (loss curves, validation accuracy vs. trainable parameters) will be saved in `results/plots/`.

### 4.2. Probabilistic Circuits
Run the following commands from the `probabilistic-circuit/` directory:
```bash
# Sync environment
uv sync

# Run the 4-bit toy correlated distribution experiment
uv run python experiments/run_toy.py

# Run the binarized MNIST density estimation experiment (CUDA supported)
uv run python experiments/run_mnist.py --width 8 --epochs 50 --max-train 1000 --max-val 200 --max-test 200
```
Additionally, `demo.ipynb` is ready to run out-of-the-box on Kaggle Notebooks utilizing a Tesla T4 GPU.

### 4.3. Data Mining: Tensor Decomposition
From the `data-mining/` directory, open and execute the Jupyter Notebooks located in `notebooks/`:
- `01_synthetic.ipynb`: Speed and convergence comparison between CP-ALS and CP-OPT on simulated data.
- `02_eeg_cp.ipynb`: Apply CP-ALS to extract trial, spatial, and temporal factors from real-world EEG signal recordings.
- `03_parafac2.ipynb`: Run PARAFAC2 on EEG datasets with irregular subject trial sizes.
- `04_cmtf.ipynb`: Fusing EEG tensor slices with static subject metadata using Coupled Matrix-Tensor Factorization.

---

## 5. LaTeX Compilation Guide
PowerShell scripts are available in the `scripts/` directory to automate LaTeX compilation via `latexmk`.

### Prerequisites
* A LaTeX distribution (e.g., TeX Live, MiKTeX) installed.
* `latexmk` executable configured in your system `PATH`.

Execute the following PowerShell commands from the repository root directory (`lab-3/`):

### 5.1. Compile the Technical Report
```powershell
scripts\compile_report.ps1
```
The compiled PDF will be generated at `report/build/main.pdf`.

### 5.2. Compile the Presentation Slides
```powershell
scripts\compile_slide.ps1
```
The compiled PDF will be generated at `slide/build/main.pdf`.

### 5.3. Project Cleanup
To clean LaTeX build files (`.aux`, `.log`, `.toc`, etc.) and Python runtime caches (`__pycache__`):
```powershell
# Preview the list of files to be cleaned without deleting them
scripts\clear_artifacts.ps1 -WhatIf

# Perform the actual cleanup
scripts\clear_artifacts.ps1
```
