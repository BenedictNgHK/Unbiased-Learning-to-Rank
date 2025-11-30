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
