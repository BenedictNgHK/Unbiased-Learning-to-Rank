# scripts/gen_logs_balanced.py  —— 新文件
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, argparse, random
from pathlib import Path
from src.searcher import SearchEngine
from src.simulation import UserSimulator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/queries.txt")
    ap.add_argument("--out", default="data/logs.more.jsonl")
    ap.add_argument("--sessions", type=int, default=12)  # 每个 query 多少次会话
    ap.add_argument("--k", type=int, default=10)         # 每次曝光多少条
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed)

    queries = [l.strip() for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    se = SearchEngine()
    sim = UserSimulator(
        click_temp=0.75,
        rel_floor=0.05,
        rel_ceiling=0.95,
        ensure_min_pos=1,
        ensure_min_neg=1,
        rng_seed=args.seed
    )

    n = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for q in queries:
            for _ in range(args.sessions):
                base = se.search_vector(q, k=args.k)   # baseline 曝光
                docs = [r["text"] for r in base]
                for L in sim.simulate_clicks(q, docs):
                    L["query"] = q
                    w.write(json.dumps(L, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Wrote {n} lines -> {args.out}")

if __name__ == "__main__":
    main()