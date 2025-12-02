from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time, random
import numpy as np
import torch

from src.searcher import SearchEngine
from src.simulation import UserSimulator
from src.evaluation import Evaluator
import src.config as config

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev 环境放开；上线请收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
search_engine = None
simulator = None
evaluator = None

@app.on_event("startup")
def load_resources():
    global search_engine, simulator, evaluator
    print("Loading resources...")
    try:
        search_engine = SearchEngine()  # 默认不强制 ULTR，按请求切换
        simulator = UserSimulator()
        evaluator = Evaluator()
        print("Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

# ------- 请求/响应模型 -------
class SearchQuery(BaseModel):
    query: str
    use_ultr: Optional[bool] = False
    seed: Optional[int] = None

class SearchResponse(BaseModel):
    results: list
    metrics: dict
    simulation_logs: list
    used_ultr: Optional[bool] = None
    seed: Optional[int] = None
    ts: int
    ultr_available: Optional[bool] = None
    metrics_ce: Optional[dict] = None
    metrics_ultr: Optional[dict] = None
    lift_ce_vs_base: Optional[dict] = None
    lift_ultr_vs_base: Optional[dict] = None
    lift_ultr_vs_ce: Optional[dict] = None

class DocumentResponse(BaseModel):
    doc_id: str
    text: str

# -------------------- 文档接口 --------------------
@app.get("/api/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System not ready")
    text = search_engine.get_document_by_id(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "text": text}

# -------------------- 搜索接口 --------------------
@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchQuery):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query
    use_ultr = bool(request.use_ultr)
    seed = request.seed
    ts = int(time.time())

    # 可复现实验：设置随机种子（影响 torch/np 的其他路径）
    if seed is not None:
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    # 1) Baseline：向量检索 + 用 baseline 排序生成点击日志（把 seed 传进去）
    baseline_results = search_engine.search_vector(query_text, k=10)
    baseline_docs = [r['text'] for r in baseline_results]
    click_logs = simulator.simulate_clicks(query_text, baseline_docs, seed=seed)

    formatted_logs = [{
        "rank": log["rank"],
        "click": log["click"],
        "propensity": log["propensity"],
        "relevance_prob": log["relevance_prob"],
        "doc_text": (log["doc_text"][:50] + "...") if log["doc_text"] else ""
    } for log in click_logs]

    # 2) 统一候选集；分别做 CE 与 ULTR 两套重排
    candidates = search_engine.search_vector(query_text, k=40)
    ce_results   = search_engine.rerank(query_text, candidates, k=10, use_ultr=False)
    ultr_results = search_engine.rerank(query_text, candidates, k=10, use_ultr=True)

    ce_docs   = [r['text'] for r in ce_results]
    ultr_docs = [r['text'] for r in ultr_results]

    # 3) 相关性与公共 IDCG（使用“有序并集”，避免 set 顺序不稳定）
    all_seen_docs = list(dict.fromkeys(baseline_docs + ce_docs + ultr_docs))
    relevance_map = simulator.get_relevance_scores(query_text, all_seen_docs)
    all_rels = sorted(relevance_map.values(), reverse=True)
    k_eval = 10
    common_idcg = sum(rel / np.log2((i + 1) + 1) for i, rel in enumerate(all_rels[:k_eval])) or 1.0

    # 4) 计算三套指标
    base_ndcg  = Evaluator.calculate_ndcg(baseline_docs, relevance_map, k=k_eval, idcg=common_idcg)
    base_snips = Evaluator.calculate_snips(baseline_docs, click_logs)

    ce_ndcg  = Evaluator.calculate_ndcg(ce_docs,  relevance_map, k=k_eval, idcg=common_idcg)
    ce_snips = Evaluator.calculate_snips(ce_docs, click_logs)

    ultr_ndcg  = Evaluator.calculate_ndcg(ultr_docs, relevance_map, k=k_eval, idcg=common_idcg)
    ultr_snips = Evaluator.calculate_snips(ultr_docs, click_logs)

    def pct_lift(a, b):
        return float(((a - b) / b * 100) if b > 0 else 0.0)

    metrics = {
        "baseline": {"ndcg": float(base_ndcg), "snips": float(base_snips)},
        "target":   {"ndcg": float(ultr_ndcg if use_ultr else ce_ndcg),
                     "snips": float(ultr_snips if use_ultr else ce_snips)},
        "lift":     {"ndcg": pct_lift(ultr_ndcg if use_ultr else ce_ndcg, base_ndcg),
                     "snips": pct_lift(ultr_snips if use_ultr else ce_snips, base_snips)}
    }

    metrics_ce   = {"ndcg": float(ce_ndcg),   "snips": float(ce_snips)}
    metrics_ultr = {"ndcg": float(ultr_ndcg), "snips": float(ultr_snips)}
    lift_ce_vs_base   = {"ndcg": pct_lift(ce_ndcg,   base_ndcg),
                         "snips": pct_lift(ce_snips, base_snips)}
    lift_ultr_vs_base = {"ndcg": pct_lift(ultr_ndcg,   base_ndcg),
                         "snips": pct_lift(ultr_snips, base_snips)}
    lift_ultr_vs_ce   = {"ndcg": pct_lift(ultr_ndcg,   ce_ndcg),
                         "snips": pct_lift(ultr_snips, ce_snips)}

    final_results = ultr_results if use_ultr else ce_results
    ultr_available = bool(search_engine.ultr_model is not None)

    return {
        "results": final_results,
        "metrics": metrics,
        "simulation_logs": formatted_logs,
        "used_ultr": use_ultr,
        "seed": seed,
        "ts": ts,
        "ultr_available": ultr_available,
        "metrics_ce": metrics_ce,
        "metrics_ultr": metrics_ultr,
        "lift_ce_vs_base": lift_ce_vs_base,
        "lift_ultr_vs_base": lift_ultr_vs_base,
        "lift_ultr_vs_ce": lift_ultr_vs_ce
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)