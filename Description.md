# Project Description: Unbiased Learning to Rank (ULTR) System

## 1. Overview
This project implements a complete **semantic search engine** capable of simulating user behavior to evaluate Unbiased Learning to Rank (ULTR) techniques. It uses a **Sentence Transformer** (Bi-Encoder) for initial retrieval and a **Cross-Encoder** for re-ranking to mitigate position bias. The system is served via a **FastAPI** backend and accessed through a modern **React** frontend.

## 2. Implemented Modules

### 2.1. Model Architecture
- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2` (Bi-Encoder) for fast vector-based candidate generation.
- **Re-Ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Cross-Encoder) for high-precision re-ranking and acting as the "Oracle" in simulations.

### 2.2. Corpus & Data
- **Dataset**: MS MARCO Passage Ranking dataset.
- **Handling**: Streaming implementation to handle large datasets efficiently without full download.

### 2.3. Indexing
- **Technology**: FAISS (Facebook AI Similarity Search).
- **Method**: `IndexFlatIP` (Inner Product) with normalized vectors for efficient Cosine Similarity search.

### 2.4. Query and Retrieval
- **Pipeline**: Vector Recall (Top-K) -> Cross-Encoder Re-ranking (Top-N).

### 2.5. Model Serving (Backend)
- **Framework**: FastAPI.
- **Endpoints**: REST API handling search queries, simulation, and metric calculation.

### 2.6. Simulation & User Behavior
- **Click Simulation**: Implements a **Position-Based Model (PBM)**.
- **Logic**: $P(\text{Click}) = P(\text{Examine} | \text{Rank}) \times P(\text{Relevance} | \text{Content})$.
- **Propensity**: Decays with rank (e.g., $1/\text{rank}^\gamma$).

### 2.7. Evaluation (OPE)
- **Offline Policy Evaluation**: Uses **SNIPS** (Self-Normalized Inverse Propensity Scoring) to estimate performance using biased historical logs.
- **Oracle Metrics**: Calculates **nDCG** (Normalized Discounted Cumulative Gain) using the Cross-Encoder as ground truth to validate OPE estimates.

## 3. Environment Setup
To replicate this environment, use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ULTR
```
