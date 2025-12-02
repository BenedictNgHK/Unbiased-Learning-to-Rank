import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json, argparse, random
import numpy as np
from src.searcher import SearchEngine
import src.config as config

def pbm_propensities(k: int, tau: float):
    ranks = np.arange(1, k + 1)
    return 1.0 / np.power(ranks, tau)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="single query string")
    ap.add_argument("--queries_file", help="a text file with one query per line")
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.7, help="position-bias exponent in PBM (1/rank^tau)")
    ap.add_argument("--out", default="data/logs.jsonl")
    args = ap.parse_args()

    if not args.query and not args.queries_file:
        raise SystemExit("Provide --query or --queries_file")

    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            pool = [ln.strip() for ln in f if ln.strip()]
    else:
        pool = [args.query]

    eng = SearchEngine(use_ultr_ce=False)  # 日志由“baseline policy”产生
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fw:
        for s in range(args.sessions):
            q = random.choice(pool)
            # baseline：仅使用向量召回（也可换成 base CE 的重排）
            cand = eng.search_vector(q, k=args.k)
            doc_ids = [c["doc_id"] for c in cand]

            # 吸引度（Oracle 转概率）
            logits = np.array(eng.oracle_score(q, doc_ids), dtype=np.float32)
            attractiveness = 1.0 / (1.0 + np.exp(-logits))

            # 位置偏置
            prop = pbm_propensities(args.k, args.tau)

            display = []
            for i, (did, att, p) in enumerate(zip(doc_ids, attractiveness, prop), start=1):
                click_prob = float(att) * float(p)
                click = int(np.random.rand() < click_prob)
                display.append({
                    "doc_id": did,
                    "rank": i,
                    "propensity": float(p),
                    "attractiveness": float(att),
                    "click": click
                })

            fw.write(json.dumps({"query": q, "policy": "vector", "display": display}, ensure_ascii=False) + "\n")

    print(f"Wrote logs to {args.out}")

if __name__ == "__main__":
    main()