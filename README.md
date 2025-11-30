# Unbiased Learning to Rank (Semantic Search MVP)

This project implements a ULTR-based semantic search system. It uses **simulated user clicks** and **Offline Policy Evaluation (OPE)** to demonstrate how to mitigate position bias and evaluate ranking performance without online testing.

## Features

1.  **Semantic Search**: Uses `sentence-transformers` and FAISS for retrieval.
2.  **Re-ranking**: Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to improve ranking quality.
3.  **Simulation**: Simulates user clicks with position bias (PBM model).
4.  **Evaluation**: Estimates performance using **SNIPS** (Self-Normalized Inverse Propensity Scoring).

## Setup

1.  **Activate your environment**:
    ```bash
    conda activate Unbiased_LTR
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Build the Index
Downloads a subset of MS MARCO and builds the FAISS index.

```bash
python main.py --index
```

### 2. Search & Evaluate
Runs the full pipeline:
1.  **Simulation**: Generates synthetic "historical" click logs from a baseline ranker.
2.  **Ranking**: Retrieval + Re-ranking (Target Policy).
3.  **Evaluation**: Compares the Target Policy against the historical logs using SNIPS.

```bash
python main.py --query "What is the impact of bias in retrieval?"
```

Or interactive mode:

```bash
python main.py --interactive
```

## Modules

- `src/simulation.py`: Simulates user behavior (clicks, propensity).
- `src/evaluation.py`: Implements OPE metrics (SNIPS).
- `src/searcher.py`: Handles Vector Search and Cross-Encoder Re-ranking.
