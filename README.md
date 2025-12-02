# Unbiased Learning to Rank (ULTR) - Search Engine

This project implements a ULTR-based search engine that simulates user behavior, re-ranks results using a Cross-Encoder, and evaluates performance using Offline Policy Evaluation (OPE) metrics like SNIPS.

It features a **FastAPI backend** for search logic and simulation, and a **React + Tailwind CSS frontend** for a modern Google-like search experience.

## Project Structure

- **Backend**: Python (FastAPI, Sentence-Transformers, FAISS)
- **Frontend**: React (Vite, Tailwind CSS, Lucide Icons)
- **Data**: MS MARCO Passage Ranking dataset (via HuggingFace)

## Prerequisites

- Conda (recommended) or Python 3.8+
- Node.js 16+ and npm

## Setup

### 1. Backend Setup

1.  **Create and Activate Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate ULTR
    ```

2.  **Build the Search Index**:
    Before running the app, you need to download data and build the FAISS index.
    ```bash
    python main.py --index
    ```
    *Note: This uses a subset of MS MARCO by default. To index more data, use `--limit N` or remove the limit in the code.*
    
3.  **prepare queries.txt**:

4.  **generate the logs.more.jsonl based on query.txt**:
    ```bash
    python scripts/gen_logs_balanced.py \
      --queries data/queries.txt \
      --sessions 80 \
      --k 10 \
      --seed 2030

    # Append new logs into the main file used for training
    cat data/logs.more.jsonl >> data/logs.jsonl
    ```
5. **train the ULTR Cross-encoder(IPS pairwise)**:
   ```bash
   python -m src.train_ultr \
      --logs data/logs.more.jsonl \
      --epochs 2 \
      --batch_size 16 \
      --neg_per_pos 3 \
      --clip 10.0 \
      --save_dir models/ce_ultr
   ```

### 2. Frontend Setup

1.  **Install Dependencies**:
    ```bash
    cd frontend/ULTR
    npm install
    ```

## Running the Application

You will need **two terminal windows** to run the full stack.

### Terminal 1: Start Backend API
```bash
# From project root
conda activate ULTR
python app.py
```
*The API will start at http://localhost:8000*

### Terminal 2: Start Frontend UI
```bash
# From frontend/ULTR directory
cd frontend/ULTR
npm run dev
```
*The UI will start at http://localhost:5173 (or similar)*

## Usage

1.  Open your browser to the frontend URL (e.g., `http://localhost:5173`).
2.  Type a query (e.g., "machine learning", "what is deep learning").
3.  choose if use ULTR or not
4.  choose if fix the random seed
5.  **View Results**: See the Top-10 documents re-ranked by the Cross-Encoder.
6.  **Check Evaluation**: Look at the sidebar to see the **Lift** in performance (Oracle nDCG and SNIPS) compared to the simulated baseline and the comparison between CE and      ULTR.
7.  **Inspect Logs**: Open the "Simulation Logs" accordion to see the raw simulated clicks that generated the baseline data.

## Key Components

- `app.py`: Main entry point for the backend API.
- `src/searcher.py`: Handles retrieval (FAISS) and re-ranking (Cross-Encoder).
- `src/simulation.py`: Simulates user clicks with position bias (PBM).
- `src/evaluation.py`: Calculates OPE metrics (SNIPS) and online metrics (nDCG).
- `frontend/ULTR/src/App.tsx`: Main React component for the search UI.
- `train_ultr.py`：train the ULTR model
- `gen_logs_balanced.py`:generate the click logs
