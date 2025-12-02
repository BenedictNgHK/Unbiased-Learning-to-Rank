import os, json, pickle, random, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
# 原来是：
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 改为：
from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    # 新路径更稳
    from transformers.optimization import get_linear_schedule_with_warmup
except Exception:
    # 旧版本兜底
    from transformers import get_linear_schedule_with_warmup  # type: ignore

from torch.optim import AdamW  # 用 PyTorch 自带的 AdamW
import src.config as config

@dataclass
class RawPair:
    query: str
    pos_doc: str
    neg_doc: str
    weight: float  # IPS weight = 1 / propensity (clipped)

class PairDataset(Dataset):
    def __init__(self, items: List[RawPair]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_fn(batch: List[RawPair], tok, max_len: int):
    q_pos = [b.query for b in batch]
    d_pos = [b.pos_doc for b in batch]
    q_neg = [b.query for b in batch]
    d_neg = [b.neg_doc for b in batch]
    w     = torch.tensor([b.weight for b in batch], dtype=torch.float32)

    # 关键：强制两边用完全一致的 padding 策略与长度
    enc_pos = tok(
        q_pos, d_pos,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors="pt",
        # pad_to_multiple_of=8,  # 若启用 AMP/GPU 对齐可打开
    )
    enc_neg = tok(
        q_neg, d_neg,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors="pt",
        # pad_to_multiple_of=8,
    )

    # 某些模型没有 token_type_ids，这里做下健壮合并
    keys = set(enc_pos.keys()) & set(enc_neg.keys())
    enc = {k: torch.cat([enc_pos[k], enc_neg[k]], dim=0) for k in keys}

    return enc, w
    

def build_pairs_from_logs(log_path: str,
                          id2text: Dict[str, str],
                          neg_per_pos: int = 4,
                          clip: float = 10.0) -> List[RawPair]:
    pairs: List[RawPair] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            q = j["query"]
            disp = j["display"]  # list of {doc_id, rank, propensity, attractiveness, click}
            pos = [d for d in disp if int(d.get("click", 0)) == 1]
            neg = [d for d in disp if int(d.get("click", 0)) == 0]
            if not pos or not neg: 
                continue
            for p in pos:
                pd = id2text.get(p["doc_id"], "")
                if not pd:
                    continue
                w = 1.0 / max(1e-6, float(p.get("propensity", 1.0)))
                if clip is not None:
                    w = min(w, float(clip))
                # 采样负例
                ns = random.sample(neg, k=min(neg_per_pos, len(neg)))
                for n in ns:
                    nd = id2text.get(n["doc_id"], "")
                    if not nd:
                        continue
                    pairs.append(RawPair(query=q, pos_doc=pd, neg_doc=nd, weight=w))
    return pairs

def train_ultr(logs_path: str,
               save_dir: str,
               base_ce: str,
               epochs: int = 1,
               lr: float = 2e-5,
               batch_size: int = 8,
               neg_per_pos: int = 4,
               clip: float = 10.0,
               max_len: int = None):
    max_len = max_len or config.MAX_LEN_CE

    # id -> text
    with open(config.DOC_IDS_FILE, "rb") as f:
        data = pickle.load(f)
    id2text = {did: txt for did, txt in zip(data["doc_ids"], data["documents"])}

    pairs = build_pairs_from_logs(logs_path, id2text, neg_per_pos=neg_per_pos, clip=clip)
    if not pairs:
        raise RuntimeError("No training pairs built from logs. Increase sessions or check log format.")

    tok = AutoTokenizer.from_pretrained(base_ce)
    model = AutoModelForSequenceClassification.from_pretrained(base_ce, num_labels=1)
    if torch.cuda.is_available():
        model.to("cuda")

    dl = DataLoader(PairDataset(pairs), batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, tok, max_len))

    opt = AdamW(model.parameters(), lr=lr)
    total_steps = len(dl) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss, tot_w, steps = 0.0, 0.0, 0
        for enc, w in dl:
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
                w = w.to("cuda")
            out = model(**enc)
            logits = out.logits.squeeze(-1)  # shape: [2B]
            B = logits.shape[0] // 2
            s_pos, s_neg = logits[:B], logits[B:]

            # RankNet（加权）：softplus(-(s_pos - s_neg)) * w
            loss = torch.nn.functional.softplus(-(s_pos - s_neg))
            loss = (loss * w).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            tot_loss += loss.item() * B
            tot_w += w.sum().item()
            steps += 1

        print(f"Epoch {ep:02d} | loss={tot_loss / max(1.0, tot_w):.6f} | steps={steps}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"Saved ULTR CE to {save_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="JSONL click logs (generated by scripts/gen_logs.py)")
    ap.add_argument("--save_dir", default=config.ULTR_MODEL_DIR)
    ap.add_argument("--base_ce",  default=config.CROSS_ENCODER_NAME)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--neg_per_pos", type=int, default=4)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--max_len", type=int, default=config.MAX_LEN_CE)
    args = ap.parse_args()

    train_ultr(args.logs, args.save_dir, args.base_ce,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
               neg_per_pos=args.neg_per_pos, clip=args.clip, max_len=args.max_len)