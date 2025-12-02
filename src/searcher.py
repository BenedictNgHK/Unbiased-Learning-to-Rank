import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import src.config as config

class SearchEngine:
    def __init__(self, use_ultr_ce: bool = False):
        self.index = None
        self.documents = []
        self.doc_ids = []
        self.model = None               # Bi-Encoder for vector search
        self.cross_encoder = None       # Base Cross-Encoder
        self.ultr_tokenizer = None      # Fine-tuned CE (optional)
        self.ultr_model = None
        self.oracle = None              # Oracle CE for simulation/OPE
        self.max_len_ce = getattr(config, "MAX_LEN_CE", 256)
        self.ultr_dir = getattr(config, "ULTR_MODEL_DIR", "models/ce_ultr")
        self._load_resources(use_ultr_ce)

    def _load_resources(self, use_ultr_ce: bool):
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
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_NAME, max_length=self.max_len_ce)

        # Oracle for simulation / OPE
        oracle_name = getattr(config, "ORACLE_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        print(f"Loading Simulator Ground Truth (Oracle): {oracle_name}...")
        self.oracle = CrossEncoder(oracle_name, max_length=self.max_len_ce)

        # Optional ULTR fine-tuned CE
        if use_ultr_ce or os.path.isdir(self.ultr_dir):
            self._try_load_ultr()

    def _try_load_ultr(self):
        if not os.path.isdir(self.ultr_dir):
            print(f"ULTR dir not found: {self.ultr_dir}")
            return
        print(f"Loading ULTR fine-tuned CE from: {self.ultr_dir}")
        self.ultr_tokenizer = AutoTokenizer.from_pretrained(self.ultr_dir)
        self.ultr_model = AutoModelForSequenceClassification.from_pretrained(self.ultr_dir)
        self.ultr_model.eval()
        if torch.cuda.is_available():
            self.ultr_model.to("cuda")

    def get_document_by_id(self, doc_id):
        """
        Retrieves a document's text by its ID. Optimized for 'doc_{index}'.
        """
        try:
            if doc_id.startswith("doc_"):
                idx = int(doc_id.split("_")[1])
                if 0 <= idx < len(self.documents):
                    return self.documents[idx]
        except Exception:
            pass
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None

    def search_vector(self, query, k=20):
        """
        Initial retrieval using vector similarity.
        """
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if 0 <= idx < len(self.documents):
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "score": float(distances[0][i]),
                    "text": self.documents[idx]
                })
        return results

    @torch.inference_mode()
    def _score_with_ultr(self, query, texts):
        if self.ultr_model is None or self.ultr_tokenizer is None:
            pairs = [[query, t] for t in texts]
            return self.cross_encoder.predict(pairs).tolist()
        toks = self.ultr_tokenizer([query]*len(texts), texts,
                                   truncation=True, padding=True,
                                   max_length=self.max_len_ce, return_tensors="pt")
        if torch.cuda.is_available():
            toks = {k: v.to("cuda") for k, v in toks.items()}
        logits = self.ultr_model(**toks).logits.squeeze(-1)
        return logits.detach().float().cpu().numpy().tolist()

    def rerank(self, query, initial_results, k=5, use_ultr=False):
        """
        Re-rank using base CE or ULTR CE.
        """
        if not initial_results:
            return []
        if use_ultr and (self.ultr_model is None):
            self._try_load_ultr()

        texts = [res["text"] for res in initial_results]
        if use_ultr:
            scores = self._score_with_ultr(query, texts)
        else:
            scores = self.cross_encoder.predict([[query, t] for t in texts]).tolist()

        reranked_results = []
        for res, score in zip(initial_results, scores):
            res = dict(res)
            res['score'] = float(score)
            reranked_results.append(res)
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        return reranked_results[:k]

    def search(self, query, k=5, use_ultr=False):
        """
        Full pipeline: Vector -> Re-ranking
        """
        candidates = self.search_vector(query, k=k*4)
        final_results = self.rerank(query, candidates, k=k, use_ultr=use_ultr)
        return final_results

    # ---- Oracle scoring for simulation/OPE/log generation ----
    def oracle_score(self, query, doc_ids):
        pairs = [(query, self.get_document_by_id(did) or "") for did in doc_ids]
        return self.oracle.predict(pairs).tolist()


if __name__ == "__main__":
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