# src/train_ultr.py  —— 统一替换整文件
import os, json, math, random, time
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- 超参 --------
LOG_FILE = os.environ.get("ULTR_LOG", "data/logs.more.jsonl")
SAVE_DIR = os.environ.get("ULTR_SAVE", "models/ce_ultr")
BASE_CE  = os.environ.get("ULTR_BASE_CE", "cross-encoder/ms-marco-MiniLM-L-6-v2")

EPOCHS = int(os.environ.get("ULTR_EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("ULTR_BS", "16"))
LR = float(os.environ.get("ULTR_LR", "2e-5"))
MAX_LEN = int(os.environ.get("ULTR_MAXLEN", "256"))
NEG_PER_POS = int(os.environ.get("ULTR_NEG_PER_POS", "3"))
SEED = int(os.environ.get("ULTR_SEED", "2025"))

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------- 工具 --------
def _to_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except:
        return default

def _norm_log(j: Dict) -> Dict:
    """容错：兼容多种字段名，统一成 {query, text, click, prop}"""
    q = j.get("query") or j.get("q") or j.get("Query")
    text = j.get("doc_text") or j.get("text") or j.get("passage") or j.get("doc")
    c = j.get("click") if "click" in j else j.get("clicked", 0)
    prop = j.get("propensity") if "propensity" in j else j.get("prop", 0.0)

    click = 1 if (c in (1, True, "1", "true", "True")) else 0
    prop = _to_float(prop, 0.0)

    return {"query": q, "text": text, "click": click, "prop": prop}

# -------- 读日志并造 pair --------
def build_pairs(log_path: str, neg_per_pos=3) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]], List[float], Dict]:
    """
    返回：(pos_pairs, neg_pairs, weights, stats)
    我们做 pairwise-IPS：对每个 query（或一个会话组），
    取所有 click=1 为正，click=0 为负；对每个正样本，随机配若干负样本；
    loss = - w * log(sigmoid(s_pos - s_neg)), 其中 w = 1 / max(prop, 1e-6)
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found")

    by_q: Dict[str, List[Dict]] = {}
    tot = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except:
                continue
            x = _norm_log(j)
            if not x["query"] or not x["text"]:
                continue
            by_q.setdefault(x["query"], []).append(x); tot += 1

    stats = {"lines": tot, "queries": len(by_q), "pairs": 0, "pos_lines":0, "neg_lines":0}

    pos_pairs: List[Tuple[str,str]] = []
    neg_pairs: List[Tuple[str,str]] = []
    weights: List[float] = []

    for q, items in by_q.items():
        pos = [it for it in items if it["click"]==1 and _to_float(it["prop"],0.0)>0]
        neg = [it for it in items if it["click"]==0]
        stats["pos_lines"] += len(pos); stats["neg_lines"] += len(neg)
        if not pos or not neg:  # 该 query 无法配对
            continue
        # 为每个正样本采样若干负样本
        for p in pos:
            w = 1.0 / max(_to_float(p["prop"], 1e-6), 1e-6)
            K = min(neg_per_pos, len(neg))
            neg_sample = random.sample(neg, K)
            for n in neg_sample:
                pos_pairs.append((q, p["text"]))
                neg_pairs.append((q, n["text"]))
                weights.append(w)

    stats["pairs"] = len(pos_pairs)
    return pos_pairs, neg_pairs, weights, stats

# -------- Dataset / Collate --------
@dataclass
class PairBatch:
    input_ids_pos: torch.Tensor
    attention_mask_pos: torch.Tensor
    input_ids_neg: torch.Tensor
    attention_mask_neg: torch.Tensor
    weights: torch.Tensor

class PairDataset(Dataset):
    def __init__(self, pos_pairs, neg_pairs, weights):
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs
        self.weights = weights
        assert len(self.pos_pairs)==len(self.neg_pairs)==len(self.weights)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        return self.pos_pairs[idx], self.neg_pairs[idx], self.weights[idx]

def make_collate(tokenizer):
    def collate(batch):
        qs_pos, ds_pos, ws = [], [], []
        qs_neg, ds_neg = [], []
        for (q_pos, d_pos), (q_neg, d_neg), w in batch:
            qs_pos.append(q_pos); ds_pos.append(d_pos); ws.append(float(w))
            qs_neg.append(q_neg); ds_neg.append(d_neg)
        enc_pos = tokenizer(qs_pos, ds_pos, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc_neg = tokenizer(qs_neg, ds_neg, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        return PairBatch(
            input_ids_pos=enc_pos["input_ids"],
            attention_mask_pos=enc_pos["attention_mask"],
            input_ids_neg=enc_neg["input_ids"],
            attention_mask_neg=enc_neg["attention_mask"],
            weights=torch.tensor(ws, dtype=torch.float32),
        )
    return collate

# -------- 训练主流程 --------
def train():
    print(f"==> Loading logs from {LOG_FILE}")
    pos_pairs, neg_pairs, weights, st = build_pairs(LOG_FILE, neg_per_pos=NEG_PER_POS)
    print(f"Queries with logs: {st['queries']}")
    print("==> Building pairwise IPS samples ...")
    print(f"Total pairs: {st['pairs']}  | pos_lines={st['pos_lines']}  neg_lines={st['neg_lines']}")
    if st["pairs"] == 0:
        raise RuntimeError("No training pairs built. Check your logs.jsonl format & fields.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_CE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_CE, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.train()

    ds = PairDataset(pos_pairs, neg_pairs, weights)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=make_collate(tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # 我们做 sigmoid(s_pos - s_neg)

    steps = 0
    for ep in range(1, EPOCHS+1):
        t0=time.time(); running=0.0
        for batch in dl:
            steps += 1
            opt.zero_grad(set_to_none=True)
            # 正/负分别前向，得到 s_pos / s_neg
            for k in ["input_ids_pos","attention_mask_pos","input_ids_neg","attention_mask_neg"]:
                pass
            enc_pos = { "input_ids": batch.input_ids_pos.to(device), "attention_mask": batch.attention_mask_pos.to(device) }
            enc_neg = { "input_ids": batch.input_ids_neg.to(device), "attention_mask": batch.attention_mask_neg.to(device) }

            s_pos = model(**enc_pos).logits.squeeze(-1)  # [B]
            s_neg = model(**enc_neg).logits.squeeze(-1)  # [B]
            s_diff = s_pos - s_neg

            # pairwise logistic: -log sigmoid(s_pos - s_neg)，乘 IPS 权重
            y = torch.ones_like(s_diff)
            loss_vec = loss_fn(s_diff, y) * batch.weights.to(device)
            loss = loss_vec.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            if steps % 50 == 0:
                print(f"Epoch {ep} | step {steps} | loss {running/50:.4f}")
                running=0.0
        print(f"Epoch {ep} done in {time.time()-t0:.1f}s")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved fine-tuned CE to: {SAVE_DIR}")

if __name__ == "__main__":
    train()