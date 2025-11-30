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
