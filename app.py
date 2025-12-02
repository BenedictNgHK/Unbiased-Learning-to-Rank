from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from src.searcher import SearchEngine
from src.simulation import UserSimulator
from src.evaluation import Evaluator
import numpy as np
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- globals ----------
search_engine: Optional[SearchEngine] = None
simulator: Optional[UserSimulator] = None
evaluator: Optional[Evaluator] = None

@app.on_event("startup")
def load_resources():
    global search_engine, simulator, evaluator
    print("Loading resources...")
    search_engine = SearchEngine()
    simulator = UserSimulator()
    evaluator = Evaluator()
    print("Resources loaded successfully.")

# ---------- models ----------
class SearchQuery(BaseModel):
    query: str
    use_ultr: Optional[bool] = False
    seed: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    simulation_logs: List[Dict[str, Any]]
    used_ultr: bool
    seed: Optional[int] = None
    ts: float
    # 新增对比字段（前端可选用）
    metrics_ce: Optional[Dict[str, float]] = None
    metrics_ultr: Optional[Dict[str, float]] = None
    lift_ce_vs_base: Optional[Dict[str, float]] = None
    lift_ultr_vs_base: Optional[Dict[str, float]] = None
    lift_ultr_vs_ce: Optional[Dict[str, float]] = None
    ultr_available: Optional[bool] = None

class DocumentResponse(BaseModel):
    doc_id: str
    text: str

# ---------- utils ----------
def _calc_idcg(sorted_rels: List[float], k: int) -> float:
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_rels[:k]))
    return idcg or 1.0

def _pack_metrics(docs: List[str],
                  relevance_map: Dict[str, float],
                  click_logs: List[Dict[str, Any]],
                  k: int,
                  idcg: float) -> Dict[str, float]:
    ndcg = evaluator.calculate_ndcg(docs, relevance_map, k=k, idcg=idcg)
    snips = evaluator.calculate_snips(docs, click_logs)
    return {"ndcg": float(ndcg), "snips": float(snips)}

def _lift(a: float, b: float) -> float:
    # (a - b)/b*100, b==0 时返回 0
    return float(((a - b) / b * 100.0) if b > 0 else 0.0)

# ---------- health ----------
@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}

# ---------- document ----------
@app.get("/api/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System not ready")
    text = search_engine.get_document_by_id(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "text": text}

# ---------- search ----------
@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchQuery):
    if not (search_engine and simulator and evaluator):
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query
    use_ultr = bool(request.use_ultr)
    seed = request.seed

    # 固定随机性（点击日志、打分中的随机）
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1) Baseline：仅向量检索 + 用户点击日志（用于 OPE/SNIPS）
    baseline_results = search_engine.search_vector(query_text, k=10)
    baseline_docs = [r['text'] for r in baseline_results]
    click_logs = simulator.simulate_clicks(query_text, baseline_docs)

    formatted_logs = [{
        "rank": log["rank"],
        "click": log["click"],
        "propensity": log["propensity"],
        "relevance_prob": log["relevance_prob"],
        "doc_text": (log["doc_text"][:50] + "...") if log["doc_text"] else ""
    } for log in click_logs]

    # 2) 生成候选，分别给 CE 与 ULTR 重排（同一批候选，便于公平对比）
    candidates = search_engine.search_vector(query_text, k=20)
    results_ce   = search_engine.rerank(query_text, candidates, k=10, use_ultr=False)
    results_ultr = search_engine.rerank(query_text, candidates, k=10, use_ultr=True)

    docs_ce   = [r['text'] for r in results_ce]
    docs_ultr = [r['text'] for r in results_ultr]

    # 3) 统一真值与 IDCG（公平对比）
    all_seen_docs = list(set(baseline_docs + docs_ce + docs_ultr))
    relevance_map = simulator.get_relevance_scores(query_text, all_seen_docs)
    all_rels = sorted(relevance_map.values(), reverse=True)
    k_eval = 10
    idcg = _calc_idcg(all_rels, k_eval)

    # 4) 三套指标
    m_base = _pack_metrics(baseline_docs, relevance_map, click_logs, k_eval, idcg)
    m_ce   = _pack_metrics(docs_ce,       relevance_map, click_logs, k_eval, idcg)
    m_ultr = _pack_metrics(docs_ultr,     relevance_map, click_logs, k_eval, idcg)

    # 5) 三种 Lift
    lift_ce_vs_base = {
        "ndcg": _lift(m_ce["ndcg"], m_base["ndcg"]),
        "snips": _lift(m_ce["snips"], m_base["snips"])
    }
    lift_ultr_vs_base = {
        "ndcg": _lift(m_ultr["ndcg"], m_base["ndcg"]),
        "snips": _lift(m_ultr["snips"], m_base["snips"])
    }
    lift_ultr_vs_ce = {
        "ndcg": _lift(m_ultr["ndcg"], m_ce["ndcg"]),
        "snips": _lift(m_ultr["snips"], m_ce["snips"])
    }

    # 向后兼容：根据 use_ultr 返回旧的 metrics/target
    chosen_results = results_ultr if use_ultr else results_ce
    chosen_docs    = docs_ultr    if use_ultr else docs_ce
    chosen_metrics = _pack_metrics(chosen_docs, relevance_map, click_logs, k_eval, idcg)
    chosen_lift = {
        "ndcg": _lift(chosen_metrics["ndcg"], m_base["ndcg"]),
        "snips": _lift(chosen_metrics["snips"], m_base["snips"])
    }
    metrics_legacy = {
        "baseline": m_base,
        "target":   chosen_metrics,
        "lift":     chosen_lift
    }

    resp = SearchResponse(
        results=chosen_results,
        metrics=metrics_legacy,           # 兼容旧前端
        simulation_logs=formatted_logs,
        used_ultr=use_ultr,
        seed=seed,
        ts=time.time(),
        metrics_ce=m_ce,
        metrics_ultr=m_ultr,
        lift_ce_vs_base=lift_ce_vs_base,
        lift_ultr_vs_base=lift_ultr_vs_base,
        lift_ultr_vs_ce=lift_ultr_vs_ce,
        ultr_available=bool(search_engine.ultr_model is not None),
    )
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)