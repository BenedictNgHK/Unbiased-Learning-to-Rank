import os

# ---- Paths ----
DATA_DIR  = "data"
INDEX_DIR = "index"
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---- Models ----
MODEL_NAME          = "sentence-transformers/all-MiniLM-L6-v2"       # Bi-Encoder
CROSS_ENCODER_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"         # Base CE
ORACLE_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"        # Oracle for simulation/OPE
MAX_LEN_CE          = 256

# ---- ULTR fine-tuned CE output dir ----
ULTR_MODEL_DIR = "models/ce_ultr"

# ---- Indexing ----
EMBEDDING_DIM = 384
INDEX_FILE    = os.path.join(INDEX_DIR, "faiss_index.bin")
DOC_IDS_FILE  = os.path.join(INDEX_DIR, "doc_ids.pkl")
BATCH_SIZE    = 64

# ---- Simulation defaults ----
DEFAULT_K = 10
DEFAULT_POSITION_BIAS_POWER = 0.5  # 1/rank^power