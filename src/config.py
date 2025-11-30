import os

# Paths
DATA_DIR = "data"
INDEX_DIR = "index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Indexing
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
DOC_IDS_FILE = os.path.join(INDEX_DIR, "doc_ids.pkl")
BATCH_SIZE = 64 # Batch size for embedding generation

# Data
# We will use a small subset for demonstration if needed
DATASET_NAME = "trec-covid" # Example from BEIR, or we can use simple sentences

# Simulation Defaults
DEFAULT_K = 10
DEFAULT_POSITION_BIAS_POWER = 0.5  # 1/rank^power
