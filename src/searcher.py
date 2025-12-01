import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import src.config as config

class SearchEngine:
    def __init__(self):
        self.index = None
        self.documents = []
        self.doc_ids = []
        self.model = None
        self.cross_encoder = None
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

        print(f"Loading Embedding model: {config.MODEL_NAME}...")
        self.model = SentenceTransformer(config.MODEL_NAME)
        
        print(f"Loading Cross-Encoder model: {config.CROSS_ENCODER_NAME}...")
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_NAME)

    def get_document_by_id(self, doc_id):
        """
        Retrieves a document's text by its ID.
        Optimized for "doc_{index}" format.
        """
        try:
            # Extract index from "doc_123"
            if doc_id.startswith("doc_"):
                idx = int(doc_id.split("_")[1])
                if 0 <= idx < len(self.documents):
                    # Verify ID match (optional, but good for safety)
                    # if self.doc_ids[idx] == doc_id: 
                    return self.documents[idx]
        except Exception:
            pass
            
        # Fallback: Linear search (slow, but safe)
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None

    def search_vector(self, query, k=20):
        """
        Performs the initial retrieval using vector similarity.
        Returns a larger pool of candidates for re-ranking.
        """
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

    def rerank(self, query, initial_results, k=5):
        """
        Re-ranks the initial results using a Cross-Encoder.
        """
        if not initial_results:
            return []
            
        # Prepare pairs [ [query, doc1], [query, doc2], ... ]
        pairs = [[query, res['text']] for res in initial_results]
        
        # Predict scores
        scores = self.cross_encoder.predict(pairs)
        
        # Update scores and sort
        reranked_results = []
        for res, score in zip(initial_results, scores):
            res['score'] = float(score) # Update with CE score
            reranked_results.append(res)
            
        # Sort by new score descending
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked_results[:k]

    def search(self, query, k=5):
        """
        Full pipeline: Vector Search -> Re-ranking
        """
        # 1. Retrieve more candidates than k (e.g., 5*k or fixed 50)
        candidates = self.search_vector(query, k=k*4)
        
        # 2. Re-rank
        final_results = self.rerank(query, candidates, k=k)
        
        return final_results

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
