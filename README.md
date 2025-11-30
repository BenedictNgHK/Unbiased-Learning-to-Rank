# Unbiased Learning to Rank (Semantic Search MVP)

This project implements a semantic search system using Sentence Transformers and FAISS, as described in the project requirements.

## Setup

1. **Activate your environment**:
   ```bash
   conda activate Unbiased_LTR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `sentence-transformers`, `faiss-cpu`, `datasets`, `pandas`, and `numpy` installed.*

## Usage

### 1. Build the Index
This step downloads a subset of the MS MARCO dataset (or uses dummy data if download fails), encodes it using `all-MiniLM-L6-v2`, and builds a FAISS index.

```bash
python main.py --index
```

### 2. Search
You can search using the CLI:

```bash
python main.py --query "What is machine learning?"
```

Or run in interactive mode:

```bash
python main.py --interactive
```

## Project Structure

- `src/config.py`: Configuration settings (model name, paths).
- `src/indexer.py`: Logic for loading data, generating embeddings, and indexing.
- `src/searcher.py`: Logic for loading the index and performing searches.
- `main.py`: Entry point CLI.
- `data/`: Directory for storing downloaded data (if any).
- `index/`: Directory for storing the FAISS index and document mappings.
