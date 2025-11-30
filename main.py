import argparse
import os
from src.indexer import build_index
from src.searcher import SearchEngine
from src.simulation import UserSimulator
from src.evaluation import Evaluator
import pandas as pd

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
            # Default to a smaller set if auto-triggering, unless specified? 
            # But user asked for entire dataset capability. 
            # If the user runs --index explicitly without limit, we do ALL.
            # If auto-triggered, maybe safer to do ALL? 
            # Let's stick to args.limit which defaults to None (All).
            
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
            
            snips_score = evaluator.calculate_snips(target_docs, click_logs)
            true_dcg = evaluator.calculate_dcg(target_docs, click_logs)
            
            print(f"\n>>> OPE Metric (SNIPS): {snips_score:.8f}")
            print(f">>> Oracle Metric (DCG):  {true_dcg:.8f}")

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
