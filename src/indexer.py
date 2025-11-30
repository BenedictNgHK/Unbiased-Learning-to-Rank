import time
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import src.config as config

def generate_msmarco_data(limit=None):
    """
    Generator that yields unique passages from the MS MARCO dataset.
    """
    print(f"Loading MS MARCO dataset (streaming, limit={limit if limit else 'ALL'})...")
    try:
        # 'ms_marco' v1.1 contains 'passages'
        # Using streaming to avoid full download
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
        
        seen_texts = set()
        count = 0
        
        # Use tqdm without total if limit is None
        pbar = tqdm(dataset, total=limit if limit else None, desc="Scanning dataset")
        
        for item in pbar:
            passages = item.get('passages', {})
            texts = passages.get('passage_text', [])
            
            for text in texts:
                if limit and count >= limit:
                    pbar.close()
                    return
                
                if text not in seen_texts:
                    seen_texts.add(text)
                    yield text
                    count += 1
                    
                    if limit and count >= limit:
                        pbar.close()
                        return
                        
    except Exception as e:
        print(f"Error loading MS MARCO: {e}")
        print("Falling back to dummy data.")
        for doc in load_dummy_data(limit):
            yield doc

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
    limit = limit if limit else 10
    return documents[:limit]

def build_index(limit=None):
    # 1. Prepare Model and Index
    print(f"Loading model: {config.MODEL_NAME}...")
    model = SentenceTransformer(config.MODEL_NAME)
    
    print("Initializing FAISS index...")
    dimension = config.EMBEDDING_DIM
    # Using Inner Product (IP) which is equivalent to Cosine Similarity for normalized vectors
    index = faiss.IndexFlatIP(dimension)
    
    documents = []
    doc_ids = []
    
    # 2. Batched Processing
    batch_docs = []
    
    print("Starting Indexing Process (Batched)...")
    start_time = time.time()
    
    # Use a progress bar for documents indexed if possible, but we generate them on fly
    count_indexed = 0
    
    for text in generate_msmarco_data(limit):
        batch_docs.append(text)
        
        if len(batch_docs) >= config.BATCH_SIZE:
            _index_batch(model, index, batch_docs, documents, doc_ids)
            count_indexed += len(batch_docs)
            if count_indexed % 1000 == 0:
                print(f"Indexed {count_indexed} documents...", end='\r')
            batch_docs = []
            
    # Process remaining
    if batch_docs:
        _index_batch(model, index, batch_docs, documents, doc_ids)
        count_indexed += len(batch_docs)

    print(f"\nEncoding finished in {time.time() - start_time:.2f}s.")
    print(f"Total Index size: {index.ntotal} vectors.")

    # 3. Save
    print(f"Saving index to {config.INDEX_FILE}...")
    faiss.write_index(index, config.INDEX_FILE)
    
    print(f"Saving doc_ids to {config.DOC_IDS_FILE}...")
    with open(config.DOC_IDS_FILE, "wb") as f:
        # Depending on size, this might need optimization (e.g. chunking), but pickle is simplest for now
        pickle.dump({"documents": documents, "doc_ids": doc_ids}, f)
    
    print("Indexing complete.")

def _index_batch(model, index, batch_docs, all_docs, all_ids):
    # Encode
    embeddings = model.encode(batch_docs, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize
    faiss.normalize_L2(embeddings)
    
    # Add to Index
    index.add(embeddings)
    
    # Update mappings
    start_id = len(all_ids)
    all_docs.extend(batch_docs)
    all_ids.extend([f"doc_{start_id + i}" for i in range(len(batch_docs))])

if __name__ == "__main__":
    # Default to a small limit if run directly, or use argparse in main
    build_index(limit=1000)
