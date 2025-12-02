import os, json, random, hashlib, torch, pickle, faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ULTR_DIR = "models/ce_ultr"
INDEX_FILE = "index/faiss_index.bin"
DOCMAP_FILE = "index/doc_ids.pkl"
MAXLEN = 256
SAMPLES = 8
CAND_K = 30

def sha1_of_tensor(t):
    b = t.detach().cpu().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()

def fingerprint(model):
    for n, p in model.named_parameters():
        if p.ndim >= 1:
            return n, p.numel(), float(p.norm().item()), sha1_of_tensor(p)
    return "NA", 0, 0.0, "NA"

def load_index():
    idx = faiss.read_index(INDEX_FILE)
    with open(DOCMAP_FILE, "rb") as f:
        data = pickle.load(f)
    docs = data["documents"]; doc_ids = data["doc_ids"]
    return idx, docs, doc_ids

def vector_candidates(biencoder, idx, docs, query, k):
    qv = biencoder.encode([query])
    qv = np.array(qv, dtype="float32"); faiss.normalize_L2(qv)
    D, I = idx.search(qv, k)
    I = I[0].tolist()
    return [(i, docs[i]) for i in I]

@torch.inference_mode()
def score_ce(tokenizer, model, query, texts):
    toks = tokenizer([query]*len(texts), texts, truncation=True, padding=True,
                     max_length=MAXLEN, return_tensors="pt")
    if torch.cuda.is_available():
        toks = {k:v.cuda() for k,v in toks.items()}
        model.cuda()
    logits = model(**toks).logits.squeeze(-1)
    return logits.detach().float().cpu().numpy().tolist()

def kendall_tau(order_a, order_b):
    pos = {d:i for i,d in enumerate(order_a)}
    pairs = 0; discord = 0
    for i in range(len(order_b)):
        for j in range(i+1, len(order_b)):
            pairs += 1
            di, dj = order_b[i], order_b[j]
            if pos[di] > pos[dj]:
                discord += 1
    if pairs == 0: return 1.0
    return 1 - 2*discord/pairs

def main():
    print("== Loading models ==")
    tok_base = AutoTokenizer.from_pretrained(BASE_CE)
    m_base = AutoModelForSequenceClassification.from_pretrained(BASE_CE)
    tok_ultr = AutoTokenizer.from_pretrained(ULTR_DIR)
    m_ultr = AutoModelForSequenceClassification.from_pretrained(ULTR_DIR)

    n_b, num_b, nrm_b, sha_b = fingerprint(m_base)
    n_u, num_u, nrm_u, sha_u = fingerprint(m_ultr)

    print(f"[BASE]  layer={n_b} num={num_b} || L2={nrm_b:.6f} || sha1={sha_b}")
    print(f"[ULTR]  layer={n_u} num={num_u} || L2={nrm_u:.6f} || sha1={sha_u}")
    print(f"Same first-layer fingerprint? -> {sha_b == sha_u}")

    print("\n== Ranking AB ==")
    idx, docs, doc_ids = load_index()
    biencoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    queries = [
        "what is position bias",
        "neural ranking models",
        "define propensity in click logs",
        "what is ndcg metric",
        "information retrieval basics",
        "how to reduce position bias",
        "inverse propensity weighting",
        "counterfactual evaluation"
    ]
    random.shuffle(queries)
    queries = queries[:SAMPLES]

    diffs = 0; taus = []; cors = []
    example = None

    for q in queries:
        cand = vector_candidates(biencoder, idx, docs, q, CAND_K)
        idxs = [i for i,_ in cand]
        texts = [t for _, t in cand]

        s_base = score_ce(tok_base, m_base, q, texts)
        s_ultr = score_ce(tok_ultr, m_ultr, q, texts)

        if len(s_base) != len(idxs) or len(s_ultr) != len(idxs):
            print(f"[warn] length mismatch on query={q}")
            continue

        # 用“分数数组的位置索引”排序，避免越界
        order_base_pos = np.argsort(np.array(s_base))[::-1].tolist()
        order_ultr_pos = np.argsort(np.array(s_ultr))[::-1].tolist()
        ord_base_ids = [doc_ids[idxs[p]] for p in order_base_pos]
        ord_ultr_ids = [doc_ids[idxs[p]] for p in order_ultr_pos]

        tau = kendall_tau(ord_base_ids, ord_ultr_ids); taus.append(tau)
        corr = float(np.corrcoef(s_base, s_ultr)[0,1]); cors.append(corr)

        if ord_base_ids[:10] != ord_ultr_ids[:10]:
            diffs += 1
            if example is None:
                example = (q,
                           ord_base_ids[:10],
                           ord_ultr_ids[:10])

    print(f"\nQueries tested: {len(queries)}")
    print(f"Top-10 order changed on {diffs}/{len(queries)} queries")
    if taus:
        print(f"mean Kendall-τ: {sum(taus)/len(taus):.4f}")
    if cors:
        print(f"mean Pearson corr(scores): {sum(cors)/len(cors):.4f}")

    if example:
        q, base10, ultr10 = example
        print("\nExample diff (first query with change):")
        print("Q:", q)
        print("BASE top10:", base10)
        print("ULTR top10:", ultr10)

if __name__ == "__main__":
    main()