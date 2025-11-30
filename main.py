import argparse
from src.indexer import build_index
from src.searcher import SearchEngine
import os

def main():
    parser = argparse.ArgumentParser(description="Semantic Search System")
    parser.add_argument("--index", action="store_true", help="Run indexing process")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # If no index exists, prompt or run indexing
    index_exists = os.path.exists("index/faiss_index.bin")
    if args.index or not index_exists:
        if not index_exists and not args.index:
            print("Index not found. Building index first...")
        build_index()

    if args.query:
        engine = SearchEngine()
        print(f"\nQuery: {args.query}")
        results = engine.search(args.query)
        for i, r in enumerate(results):
            print(f"{i+1}. [{r['score']:.4f}] {r['text'][:300]}...")
            
    if args.interactive:
        engine = SearchEngine()
        print("\nSearch Engine Ready. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            results = engine.search(query)
            print(f"\nTop {len(results)} results:")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text'][:300]}...")

    if not args.index and not args.query and not args.interactive:
        print("Usage:")
        print("  python main.py --index          # Build index")
        print("  python main.py --query 'text'   # Search")
        print("  python main.py --interactive    # Interactive mode")

if __name__ == "__main__":
    main()
