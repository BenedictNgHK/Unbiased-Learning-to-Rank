1. **Main Idea**:  
   Use a pre-trained Sentence Transformer to encode both documents and queries into dense vectors, then retrieve the most semantically similar documents to a given query via cosine similarity—essentially building a minimal viable system for **semantic search**. Because this core task is lightweight, it can be extended further: wrap the model as a backend API for frontend consumption, collect user interaction logs from the frontend, and leverage those logs for bias-aware training, etc.

2. **Project Modules**:

   2.1. **Model Selection**:
   - `sentence-transformers/all-MiniLM-L6-v2`
   - `BAAI/bge-small-en` or `BAAI/bge-large-en-v1.5`

   2.2. **Corpus / Evaluation Datasets**:
   - MS MARCO passages
   - BEIR benchmark

   2.3. **Indexing Method**:
   - FAISS

   2.4. **Query and Retrieval**

   2.5. **Model Serving & Log Collection** *(future extension)*

   2.6. **Click Log-Based Re-ranking with Debiasing** *(future extension)*

   2.7. **Offline Counterfactual Performance Estimation (OPE)** *(future extension)*