# Complete project with sentence-transformers
_Exported on 11/30/2025 at 17:51:29 GMT+8 from Cursor (2.1.39)_

---

**User**

@Description.md I've alread created a conda environment Unbiased_LTR and downloaded sentence-transformers via ` pip install -U sentence-transformers`. Please help me to complete thsi project.

---

**Cursor**

```pip-requirements
sentence-transformers
faiss-cpu
datasets
pandas
numpy
tqdm

```

```python
import os

# Paths
DATA_DIR = "data"
INDEX_DIR = "index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Indexing
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
DOC_IDS_FILE = os.path.join(INDEX_DIR, "doc_ids.pkl")

# Data
# We will use a small subset for demonstration if needed
DATASET_NAME = "trec-covid" # Example from BEIR, or we can use simple sentences

```

```python
import time
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import src.config as config

def load_msmarco_data(limit=1000):
    """
    Loads a subset of MS MARCO passages.
    """
    print(f"Loading MS MARCO dataset (streaming, limit={limit})...")
    try:
        # 'ms_marco' v1.1 contains 'passages'
        # Using streaming to avoid full download
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
        
        documents = []
        doc_ids = []
        
        count = 0
        for item in tqdm(dataset, total=limit):
            # Item structure: {'query_id': ..., 'query_type': ..., 'query': ..., 'passages': {'is_selected': [...], 'url': [...], 'passage_text': [...]}}
            # This dataset seems to be Q&A. We need the corpus.
            # Let's look for unique passages.
            
            passages = item.get('passages', {})
            texts = passages.get('passage_text', [])
            
            for text in texts:
                if count >= limit:
                    break
                if text not in documents: # Simple dedup for this small batch
                    documents.append(text)
                    doc_ids.append(f"doc_{count}")
                    count += 1
            if count >= limit:
                break
                
        return documents, doc_ids
    except Exception as e:
        print(f"Error loading MS MARCO: {e}")
        print("Falling back to dummy data.")
        return load_dummy_data(limit)

def load_dummy_data(limit=10):
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Sentence transformers are useful for semantic search.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Python is a high-level, general-purpose programming language.",
        "The capital of France is Paris.",
        "Semantic search seeks to improve search accuracy by understanding the searcher's intent.",
        "Biases in data can lead to biased machine learning models."
    ]
    return documents[:limit], [f"doc_{i}" for i in range(len(documents[:limit]))]

def build_index():
    # 1. Load Data
    documents, doc_ids = load_msmarco_data(limit=1000)
    if not documents:
        print("No documents loaded.")
        return

    # 2. Load Model
    print(f"Loading model: {config.MODEL_NAME}...")
    model = SentenceTransformer(config.MODEL_NAME)

    # 3. Encode
    print("Encoding documents...")
    start_time = time.time()
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    print(f"Encoding finished in {time.time() - start_time:.2f}s. Shape: {embeddings.shape}")

    # 4. Build FAISS Index
    print("Building FAISS index...")
    dimension = config.EMBEDDING_DIM
    
    # Normalize for cosine similarity if using Inner Product, or just use L2 if the model is not normalized?
    # all-MiniLM-L6-v2 produces normalized embeddings? Usually yes.
    # If normalized, L2 distance is related to cosine similarity. 
    # Or we can use faiss.IndexFlatIP for Inner Product (Cosine Similarity on normalized vectors).
    # config.py says cosine similarity in description.
    
    # FAISS IndexFlatIP is for inner product. If vectors are normalized, IP == Cosine Similarity.
    index = faiss.IndexFlatIP(dimension)
    
    # Verify normalization
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"Index contains {index.ntotal} vectors.")

    # 5. Save
    print(f"Saving index to {config.INDEX_FILE}...")
    faiss.write_index(index, config.INDEX_FILE)
    
    print(f"Saving doc_ids to {config.DOC_IDS_FILE}...")
    with open(config.DOC_IDS_FILE, "wb") as f:
        pickle.dump({"documents": documents, "doc_ids": doc_ids}, f)
    
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()

```

```python
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import src.config as config

class SearchEngine:
    def __init__(self):
        self.index = None
        self.documents = []
        self.doc_ids = []
        self.model = None
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(config.INDEX_FILE) or not os.path.exists(config.DOC_IDS_FILE):
            raise FileNotFoundError("Index or Doc IDs file not found. Run indexer.py first.")

        print("Loading FAISS index...")
        self.index = faiss.read_index(config.INDEX_FILE)

        print("Loading document mappings...")
        with open(config.DOC_IDS_FILE, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]

        print(f"Loading model: {config.MODEL_NAME}...")
        self.model = SentenceTransformer(config.MODEL_NAME)

    def search(self, query, k=5):
        # Encode query
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)

        # Search
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx < len(self.documents):
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "score": float(distances[0][i]),
                    "text": self.documents[idx]
                })
        
        return results

if __name__ == "__main__":
    # Interactive testing
    try:
        engine = SearchEngine()
        print("\nSearch Engine Ready. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ")
            if query.strip().lower() == 'exit':
                break
            
            results = engine.search(query)
            print(f"\nTop {len(results)} results:")
            for r in results:
                print(f"[{r['score']:.4f}] {r['text'][:200]}...")
    except Exception as e:
        print(f"Error: {e}")

```

```python
import argparse
from src.indexer import build_index
from src.searcher import SearchEngine
import os

def main():
    parser = argparse.ArgumentParser(description="Semantic Search System")
    parser.add_argument("--index", action="store_true", help="Run indexing process")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # If no index exists, prompt or run indexing
    index_exists = os.path.exists("index/faiss_index.bin")
    if args.index or not index_exists:
        if not index_exists and not args.index:
            print("Index not found. Building index first...")
        build_index()

    if args.query:
        engine = SearchEngine()
        print(f"\nQuery: {args.query}")
        results = engine.search(args.query)
        for i, r in enumerate(results):
            print(f"{i+1}. [{r['score']:.4f}] {r['text'][:300]}...")
            
    if args.interactive:
        engine = SearchEngine()
        print("\nSearch Engine Ready. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            results = engine.search(query)
            print(f"\nTop {len(results)} results:")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text'][:300]}...")

    if not args.index and not args.query and not args.interactive:
        print("Usage:")
        print("  python main.py --index          # Build index")
        print("  python main.py --query 'text'   # Search")
        print("  python main.py --interactive    # Interactive mode")

if __name__ == "__main__":
    main()

```

```markdown
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

```

I've set up the complete semantic search project structure for you. It implements the "Minimal Viable System" described in your `Description.md` (Modules 2.1 - 2.4), including data loading (MS MARCO subset), embedding generation, FAISS indexing, and a search interface.

### Project Components Created

1.  **`src/config.py`**: Central configuration for model names (`all-MiniLM-L6-v2`), paths, and parameters.
2.  **`src/indexer.py`**: Handles data loading (streams 1,000 MS MARCO passages to avoid large downloads), generates embeddings, and builds the FAISS index.
3.  **`src/searcher.py`**: Loads the saved index and model to perform semantic searches.
4.  **`main.py`**: A command-line interface (CLI) to run indexing and search.
5.  **`requirements.txt`**: Lists necessary libraries (`faiss-cpu`, `datasets`, etc.).
6.  **`README.md`**: Instructions for setting up and running the project.

### How to Run

Since you are in your `Unbiased_LTR` environment, follow these steps:

1.  **Install additional dependencies** (besides `sentence-transformers`):
    ```bash
    pip install -r requirements.txt
    ```

2.  **Build the Index** (this streams a small subset of data and creates the index):
    ```bash
    python main.py --index
    ```

3.  **Run a Search**:
    ```bash
    python main.py --query "What is the impact of bias in retrieval?"
    ```

    Or run interactive mode:
    ```bash
    python main.py --interactive
    ```

The system defaults to using a streaming subset of **MS MARCO** (first 1,000 passages) to keep it lightweight. If the dataset fails to load (e.g., due to network issues), it gracefully falls back to a built-in dummy dataset so you can still verify the pipeline works.

```shellscript
    pip install -r requirements.txt
```

```shellscript
    python main.py --index
```

```shellscript
    python main.py --query "What is the impact of bias in retrieval?"
```

```shellscript
    python main.py --interactive
```

