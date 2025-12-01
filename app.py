from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.searcher import SearchEngine
from src.simulation import UserSimulator
from src.evaluation import Evaluator
import src.config as config
import numpy as np
import pandas as pd

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Instances (loaded on startup)
search_engine = None
simulator = None
evaluator = None

@app.on_event("startup")
def load_resources():
    global search_engine, simulator, evaluator
    print("Loading resources...")
    try:
        search_engine = SearchEngine()
        simulator = UserSimulator()
        evaluator = Evaluator()
        print("Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

class SearchQuery(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: list
    metrics: dict
    simulation_logs: list

class DocumentResponse(BaseModel):
    doc_id: str
    text: str

@app.get("/api/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System not ready")
    
    text = search_engine.get_document_by_id(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return {"doc_id": doc_id, "text": text}

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchQuery):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query
    
    # 1. Simulation (Baseline)
    baseline_results = search_engine.search_vector(query_text, k=10)
    baseline_docs = [r['text'] for r in baseline_results]
    
    # Simulate clicks
    click_logs = simulator.simulate_clicks(query_text, baseline_docs)
    
    # Format logs for frontend
    formatted_logs = []
    for log in click_logs:
        formatted_logs.append({
            "rank": log["rank"],
            "click": log["click"],
            "propensity": log["propensity"],
            "relevance_prob": log["relevance_prob"],
            "doc_text": log["doc_text"][:50] + "..." # Truncate for log display
        })

    # 2. Ranking (ULTR Target)
    candidates = search_engine.search_vector(query_text, k=20)
    target_results = search_engine.rerank(query_text, candidates, k=10)
    target_docs = [r['text'] for r in target_results]

    # 3. Evaluation
    all_seen_docs = list(set(baseline_docs + target_docs))
    relevance_map = simulator.get_relevance_scores(query_text, all_seen_docs)
    
    # Calculate Common IDCG
    all_relevances = sorted(relevance_map.values(), reverse=True)
    k_eval = 10
    common_idcg = 0.0
    for i, rel in enumerate(all_relevances[:k_eval]):
        common_idcg += rel / np.log2((i + 1) + 1)
    if common_idcg == 0: common_idcg = 1.0

    # Calculate Metrics
    target_snips = evaluator.calculate_snips(target_docs, click_logs)
    target_ndcg = evaluator.calculate_ndcg(target_docs, relevance_map, k=k_eval, idcg=common_idcg)
    
    baseline_snips = evaluator.calculate_snips(baseline_docs, click_logs)
    baseline_ndcg = evaluator.calculate_ndcg(baseline_docs, relevance_map, k=k_eval, idcg=common_idcg)
    
    # Calculate Lift
    ndcg_lift = ((target_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0.0
    snips_lift = ((target_snips - baseline_snips) / baseline_snips * 100) if baseline_snips > 0 else 0.0

    metrics = {
        "baseline": {
            "ndcg": float(baseline_ndcg),
            "snips": float(baseline_snips)
        },
        "target": {
            "ndcg": float(target_ndcg),
            "snips": float(target_snips)
        },
        "lift": {
            "ndcg": float(ndcg_lift),
            "snips": float(snips_lift)
        }
    }

    return {
        "results": target_results,
        "metrics": metrics,
        "simulation_logs": formatted_logs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
