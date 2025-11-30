import argparse
import os
from src.indexer import build_index
from src.searcher import SearchEngine
from src.simulation import UserSimulator
from src.evaluation import Evaluator
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="ULTR Semantic Search System with OPE")
    parser.add_argument("--index", action="store_true", help="Run indexing process")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of documents to index (default: None = All)")
    args = parser.parse_args()

    # 1. Indexing Check
    index_exists = os.path.exists("index/faiss_index.bin")
    if args.index or not index_exists:
        if not index_exists and not args.index:
            print("Index not found. Building index first...")
            
        print(f"Building index with limit={args.limit if args.limit is not None else 'ALL'}...")
        build_index(limit=args.limit)

    # 2. Initialization
    if args.query or args.interactive:
        print("Initializing Search Engine and Simulator...")
        try:
            search_engine = SearchEngine()
            simulator = UserSimulator()
            evaluator = Evaluator()
        except Exception as e:
            print(f"Initialization Error: {e}")
            return

        def process_query(query_text):
            print(f"\n{'='*50}")
            print(f"Query: {query_text}")
            print(f"{'='*50}")

            # --- Step A: Simulation (The "Past") ---
            print("\n[1. Simulation] Generating historical click logs (Baseline Policy)...")
            baseline_results = search_engine.search_vector(query_text, k=10)
            
            if not baseline_results:
                print("No results found.")
                return

            baseline_docs = [r['text'] for r in baseline_results]
            
            # Simulate user interactions (Clicks)
            click_logs = simulator.simulate_clicks(query_text, baseline_docs)
            
            # Display Logs
            df_logs = pd.DataFrame(click_logs)
            print(f"Generated {len(click_logs)} log entries.")
            if not df_logs.empty:
                print(df_logs[['rank', 'click', 'propensity', 'relevance_prob']].to_string(index=False))
            
            # --- Step B: New Ranking (The "ULTR" System) ---
            print("\n[2. Ranking] Running ULTR System (Vector + Cross-Encoder)...")
            candidates = search_engine.search_vector(query_text, k=20) 
            target_results = search_engine.rerank(query_text, candidates, k=10)
            target_docs = [r['text'] for r in target_results]
            
            # Display New Ranking
            print("\nTop 5 Results:")
            for i, r in enumerate(target_results[:5]):
                print(f"{i+1}. [{r['score']:.4f}] {r['text'][:100]}...")

            # --- Step C: OPE Evaluation ---
            print("\n[3. Evaluation] Calculating OPE Metrics (Target vs Baseline Logs)...")
            
            # 1. Prepare "Ground Truth" Relevance Map for Oracle Metrics
            # We define the pool as the union of all documents seen in Baseline and Target
            all_seen_docs = list(set(baseline_docs + target_docs))
            relevance_map = simulator.get_relevance_scores(query_text, all_seen_docs)
            
            # Calculate Common IDCG (Ideal DCG of the best k documents from the union pool)
            # This ensures a fair comparison between Baseline and Target
            all_relevances = sorted(relevance_map.values(), reverse=True)
            k_eval = 10
            common_idcg = 0.0
            for i, rel in enumerate(all_relevances[:k_eval]):
                common_idcg += rel / np.log2((i + 1) + 1)
            
            if common_idcg == 0:
                common_idcg = 1.0 # Avoid div by zero if all are 0

            # 2. Calculate Metrics
            target_snips = evaluator.calculate_snips(target_docs, click_logs)
            target_ndcg = evaluator.calculate_ndcg(target_docs, relevance_map, k=k_eval, idcg=common_idcg)
            
            baseline_snips = evaluator.calculate_snips(baseline_docs, click_logs)
            baseline_ndcg = evaluator.calculate_ndcg(baseline_docs, relevance_map, k=k_eval, idcg=common_idcg)

            print(f"\n{'-'*20} Results {'-'*20}")
            print(f"Metric\t\tBaseline\tTarget\t\tLift")
            print(f"{'-'*58}")
            
            # Calculate Lift
            ndcg_lift = ((target_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0.0
            snips_lift = ((target_snips - baseline_snips) / baseline_snips * 100) if baseline_snips > 0 else 0.0
            
            print(f"Oracle nDCG\t{baseline_ndcg:.4f}\t\t{target_ndcg:.4f}\t\t{ndcg_lift:+.2f}%")
            print(f"OPE (SNIPS)\t{baseline_snips:.4f}\t\t{target_snips:.4f}\t\t{snips_lift:+.2f}%")

        if args.query:
            process_query(args.query)
            
        if args.interactive:
            print("\nULTR System Ready. Type 'exit' to quit.")
            while True:
                query = input("\nEnter query: ")
                if query.strip().lower() in ['exit', 'quit']:
                    break
                process_query(query)

    if not args.index and not args.query and not args.interactive:
        print("Usage:")
        print("  python main.py --index [--limit N]   # Build index (N=None for all)")
        print("  python main.py --query 'text'        # Search & Evaluate")
        print("  python main.py --interactive         # Interactive mode")

if __name__ == "__main__":
    main()
