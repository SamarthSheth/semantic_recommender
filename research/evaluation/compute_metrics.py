import json
import math
import os
import sys
from pathlib import Path
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend"))

from config_loader import load_config

"""
Metrics:
  - NDCG@10  (Normalized Discounted Cumulative Gain — the gold standard)
  - MRR@10   (Mean Reciprocal Rank — how early is the first good hit?)
  - P@10     (Precision — what fraction of top-10 is relevant?)
  - Recall@10 (what fraction of known-relevant movies appear in top-10?)
  - MAP@10   (Mean Average Precision — rewards clustering hits at the top)
  - Hit Rate (did we get at least one relevant result?)

Relevance grading:
  - bullseye = 2  (exactly the right movie)
  - good     = 1  (reasonable match)
  - miss     = 0

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python research/evaluation/compute_metrics.py
    
    # Without reranker (no API key needed, just baseline + enriched):
    python research/evaluation/compute_metrics.py --no-rerank
"""

TEST_QUERIES = [
    {
        "query": "a dark psychological thriller with an unreliable narrator",
        "type": "genre+concept",
        "bullseye": ["Fight Club", "Shutter Island", "Gone Girl", "Black Swan", "Memento"],
        "good": ["Sixth Sense", "Identity", "Donnie Darko", "Mulholland Drive",
                 "A Beautiful Mind", "Secret Window", "Primal Fear", "Angel Heart",
                 "Jacobs Ladder", "Perfect Blue", "Fractured", "Copycat",
                 "Images", "Repulsion", "Psycho"],
    },
    {
        "query": "visually stunning sci-fi epic about space exploration",
        "type": "style+genre",
        "bullseye": ["Interstellar", "2001", "Gravity", "Ad Astra", "Arrival"],
        "good": ["The Martian", "Prometheus", "Sunshine", "Moon", "Solaris",
                 "Star Trek", "Mission to Mars", "Contact", "Europa Report",
                 "Passengers", "First Man", "Apollo 13", "Alien"],
    },
    {
        "query": "something that feels like a rainy Sunday afternoon",
        "type": "vibes",
        "bullseye": ["Lost in Translation", "Her", "Eternal Sunshine", "In the Mood for Love", "Perfect Blue", "Aftersun"],
        "good": ["Amelie", "Paterson", "Columbus", "After Life", "Garden State",
                 "Before Sunrise", "Chungking Express", "The Secret Life of Walter Mitty",
                 "Moonrise Kingdom", "Submarine", "Frances Ha", "500 Days of Summer",
                 "Garden of Words", "Whisper of the Heart", "Only Yesterday",
                 "The Grand Budapest Hotel", "Castaway on the Moon"],
    },
    {
        "query": "fast-paced heist movie with clever twists",
        "type": "pace+genre",
        "bullseye": ["Ocean's Eleven", "The Italian Job", "Inside Man", "Baby Driver"],
        "good": ["Heat", "The Usual Suspects", "Now You See Me", "The Town",
                 "Logan Lucky", "The Score", "Snatch", "Lock, Stock",
                 "Foolproof", "Getaway", "Rififi", "Thief", "American Animals",
                 "Triple Frontier", "Asphalt Jungle", "Thunderbolt and Lightfoot"],
    },
    {
        "query": "like Blade Runner meets Her",
        "type": "comparative",
        "bullseye": ["Ex Machina", "Ghost in the Shell", "A.I."],
        "good": ["Solaris", "Eternal Sunshine", "Transcendence", "The Matrix",
                 "eXistenZ", "Electric Dreams", "Surrogates", "Blade Runner",
                 "Blade Runner 2049", "Zero Theorem", "Sleep Dealer",
                 "Bicentennial Man", "Chappie", "I, Robot", "Automata"],
    },
    {
        "query": "that anxious feeling when everything is about to fall apart",
        "type": "abstract_emotion",
        "bullseye": ["Uncut Gems", "Requiem for a Dream", "Black Swan", "Whiplash"],
        "good": ["Take Shelter", "Climax", "Hereditary", "Mother!", "Melancholia",
                 "It Comes at Night", "Unsane", "Nightcrawler", "Gone Girl",
                 "A Serious Man", "Falling Down", "Panic Room", "Prisoners",
                 "Nocturnal Animals", "Parasite"],
    },
    {
        "query": "a movie that will make me ugly cry",
        "type": "emotional",
        "bullseye": ["Schindler's List", "Grave of the Fireflies", "Life Is Beautiful"],
        "good": ["The Green Mile", "Up", "Toy Story 3", "Coco", "Marley & Me",
                 "My Girl", "Steel Magnolias", "Terms of Endearment",
                 "The Fault in Our Stars", "P.S. I Love You", "A Walk to Remember",
                 "Dear Zachary", "Hachi", "The Notebook", "Manchester by the Sea",
                 "Million Dollar Baby", "Philadelphia"],
    },
    {
        "query": "weird surreal film that messes with your head",
        "type": "style+feeling",
        "bullseye": ["Mulholland Drive", "Eraserhead", "Donnie Darko", "Being John Malkovich"],
        "good": ["Synecdoche, New York", "Naked Lunch", "The Lobster", "Brazil",
                 "Videodrome", "Jacob's Ladder", "Vanilla Sky", "Paprika",
                 "Enter the Void", "Holy Motors", "Phantom of Liberty",
                 "Inland Empire", "Waking Life", "Pi", "Tetsuo"],
    },
    {
        "query": "cozy nostalgic movie from the 80s or 90s",
        "type": "mood+era",
        "bullseye": ["The Princess Bride", "Back to the Future", "E.T.", "The Goonies"],
        "good": ["Home Alone", "Ferris Bueller", "The Breakfast Club", "Ghostbusters",
                 "Stand by Me", "The NeverEnding Story", "Labyrinth", "Big",
                 "Honey, I Shrunk the Kids", "Mrs. Doubtfire", "Hook",
                 "The Sandlot", "Jumanji", "Matilda", "Beetlejuice"],
    },
    {
        "query": "something dark and atmospheric with amazing cinematography",
        "type": "style+mood",
        "bullseye": ["Blade Runner 2049", "The Revenant", "Sicario", "No Country for Old Men"],
        "good": ["There Will Be Blood", "Apocalypse Now", "The Assassination of Jesse James",
                 "Zodiac", "Se7en", "Road to Perdition", "Prisoners",
                 "Suspiria", "Mandy", "The Witch", "Crimson Peak",
                 "Only God Forgives", "Annihilation", "Color Out of Space"],
    },
]

# ___metrics_____

def get_relevance(title: str, bullseye: list, good: list) -> int:
    title_lower = title.lower()
    for b in bullseye:
        if b.lower() in title_lower:
            return 2
    for g in good:
        if g.lower() in title_lower:
            return 1
    return 0

#discounted cumulative gain at k
def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

def ndcg_at_k(relevances: list[int], bullseye: list, good: list, k: int) -> float:
    actual_dcg = dcg_at_k(relevances, k)
    
    # Ideal: all bullseyes first (rel=2), then goods (rel=1)
    ideal_rels = sorted([2] * len(bullseye) + [1] * len(good), reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)
    
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def mrr_at_k(relevances: list[int], k: int) -> float:
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1 / (i + 1)
    return 0.0

def precision_at_k(relevances: list[int], k: int) -> float:
    hits = sum(1 for r in relevances[:k] if r > 0)
    return hits / k

def recall_at_k(relevances: list[int], bullseye: list, good: list, k: int) -> float:
    relevant_items = len(bullseye) + len(good)
    if relevant_items == 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r > 0)
    return hits / relevant_items

def average_precision_at_k(relevances: list[int], k: int) -> float:
    hits = 0
    sum_precisions = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            hits += 1
            sum_precisions += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precisions / hits

def hit_rate_at_k(relevances: list[int], k: int) -> float:
    return 1.0 if any(r > 0 for r in relevances[:k]) else 0.0

#---serarching-----
def search_faiss(model, index, movie_ids, query, top_k = 50):
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({"tmdb_id": movie_ids[idx], "score": float(score)})
    return results

RERANKER_PROMPT = """You are a movie recommendation expert. Given the user's query and candidate movies, select the 10 BEST matches ranked from most to least relevant.

Consider plot, themes, tone, mood, atmosphere, pacing, and emotional impact. Read between the lines — "unreliable narrator" means look for twist endings and dual identities even if those exact words aren't used.

Respond with ONLY a JSON array of movie numbers (1-indexed) in order of relevance.
Example: [14, 3, 27, 8, 41, 19, 2, 33, 11, 6]"""

def rerank_with_llm(query, candidates, client):
    lines = [f'User query: "{query}"\n\nCandidate movies:\n']
    for i, c in enumerate(candidates, 1):
        doc = c["document"][:300]
        lines.append(f"{i}. {c['title']}: {doc}")
    lines.append(f"\nSelect the 10 best matches from the {len(candidates)} candidates above.")
    prompt = "\n".join(lines)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=RERANKER_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        indices = json.loads(text)
        return [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(candidates)]
    except json.JSONDecodeError:
        import re
        numbers = re.findall(r'\d+', text)
        return [int(n) - 1 for n in numbers[:10] if 1 <= int(n) <= len(candidates)]
    
 #---main evaluation loop---
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--no-rerank", action="store_true", help="Skip LLM reranking (no API key needed)")
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])
    embeddings_dir = Path(config["paths"]["embeddings"])

    # Load model + index
    print("Loading model and index...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    with open(embeddings_dir / "movie_ids.json") as f:
        movie_ids = json.load(f)

    movies = pd.read_parquet(processed_dir / "movies_final.parquet")
    id_to_title = {}
    id_to_doc = {}
    for _, row in movies.iterrows():
        tid = row["tmdbId"]
        id_to_title[tid] = row.get("title", row.get("title_tmdb", "Unknown"))
        id_to_doc[tid] = row["document"]

    # Setup reranker
    client = None
    if not args.no_rerank:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("LLM reranking enabled.\n")
        else:
            print("No ANTHROPIC_API_KEY — skipping reranking.\n")

    # Setup expansion
    expand_fn = None
    if client:
        try:
            from query_expansion import expand_query
            expand_fn = expand_query
            print("Query expansion enabled.\n")
        except ImportError:
            pass

    # ── Define pipeline variants ──────────────────────────────────────
    K = args.top_k

    # Collect per-query metrics for each approach
    approaches = ["BASELINE"]
    if client:
        approaches.append("RERANKED")
    if client and expand_fn:
        approaches.append("EXPAND+RERANK")

    all_metrics = {name: {
        "ndcg": [], "mrr": [], "precision": [], "recall": [],
        "ap": [], "hit_rate": [],
    } for name in approaches}

    per_query_results = []

    for tq in TEST_QUERIES:
        query = tq["query"]
        bullseye = tq["bullseye"]
        good = tq["good"]

        print("=" * 90)
        print(f'QUERY: "{query}"  [{tq["type"]}]')
        print("=" * 90)

        # ── BASELINE: raw FAISS top-K ─────────────────────────────────
        raw = search_faiss(st_model, index, movie_ids, query, args.retrieve_k)
        baseline_rels = [get_relevance(id_to_title.get(r["tmdb_id"], ""), bullseye, good)
                         for r in raw[:K]]

        all_metrics["BASELINE"]["ndcg"].append(ndcg_at_k(baseline_rels, bullseye, good, K))
        all_metrics["BASELINE"]["mrr"].append(mrr_at_k(baseline_rels, K))
        all_metrics["BASELINE"]["precision"].append(precision_at_k(baseline_rels, K))
        all_metrics["BASELINE"]["recall"].append(recall_at_k(baseline_rels, bullseye, good, K))
        all_metrics["BASELINE"]["ap"].append(average_precision_at_k(baseline_rels, K))
        all_metrics["BASELINE"]["hit_rate"].append(hit_rate_at_k(baseline_rels, K))

        print(f"\n  BASELINE top-{K}:")
        for i, r in enumerate(raw[:K]):
            title = id_to_title.get(r["tmdb_id"], "?")
            rel = baseline_rels[i]
            marker = " ★" if rel == 2 else " ●" if rel == 1 else ""
            print(f"    {i+1:>2}. [{r['score']:.3f}] {title[:45]}{marker}")

        # ── RERANKED: LLM rerank top-50 ──────────────────────────────
        if client:
            candidates = [{"tmdb_id": r["tmdb_id"],
                          "title": id_to_title.get(r["tmdb_id"], "Unknown"),
                          "document": id_to_doc.get(r["tmdb_id"], "")}
                         for r in raw]
            try:
                reranked_idx = rerank_with_llm(query, candidates, client)
                reranked_results = [raw[i] for i in reranked_idx[:K] if i < len(raw)]
            except Exception as e:
                print(f"  Rerank failed: {e}")
                reranked_results = raw[:K]

            reranked_rels = [get_relevance(id_to_title.get(r["tmdb_id"], ""), bullseye, good)
                            for r in reranked_results]
            # Pad if reranker returned fewer than K
            while len(reranked_rels) < K:
                reranked_rels.append(0)

            all_metrics["RERANKED"]["ndcg"].append(ndcg_at_k(reranked_rels, bullseye, good, K))
            all_metrics["RERANKED"]["mrr"].append(mrr_at_k(reranked_rels, K))
            all_metrics["RERANKED"]["precision"].append(precision_at_k(reranked_rels, K))
            all_metrics["RERANKED"]["recall"].append(recall_at_k(reranked_rels, bullseye, good, K))
            all_metrics["RERANKED"]["ap"].append(average_precision_at_k(reranked_rels, K))
            all_metrics["RERANKED"]["hit_rate"].append(hit_rate_at_k(reranked_rels, K))

            print(f"\n  RERANKED top-{K}:")
            for i, r in enumerate(reranked_results[:K]):
                title = id_to_title.get(r["tmdb_id"], "?")
                rel = reranked_rels[i]
                marker = " ★" if rel == 2 else " ●" if rel == 1 else ""
                print(f"    {i+1:>2}. {title[:45]}{marker}")

            time.sleep(0.3)

        # ── EXPAND + RERANK ──────────────────────────────────────────
        if client and expand_fn:
            expanded = expand_fn(query)
            exp_raw = search_faiss(st_model, index, movie_ids, expanded, args.retrieve_k)
            exp_candidates = [{"tmdb_id": r["tmdb_id"],
                              "title": id_to_title.get(r["tmdb_id"], "Unknown"),
                              "document": id_to_doc.get(r["tmdb_id"], "")}
                             for r in exp_raw]
            try:
                exp_idx = rerank_with_llm(query, exp_candidates, client)
                exp_results = [exp_raw[i] for i in exp_idx[:K] if i < len(exp_raw)]
            except Exception as e:
                print(f"  Expand+rerank failed: {e}")
                exp_results = exp_raw[:K]

            exp_rels = [get_relevance(id_to_title.get(r["tmdb_id"], ""), bullseye, good)
                       for r in exp_results]
            while len(exp_rels) < K:
                exp_rels.append(0)

            all_metrics["EXPAND+RERANK"]["ndcg"].append(ndcg_at_k(exp_rels, bullseye, good, K))
            all_metrics["EXPAND+RERANK"]["mrr"].append(mrr_at_k(exp_rels, K))
            all_metrics["EXPAND+RERANK"]["precision"].append(precision_at_k(exp_rels, K))
            all_metrics["EXPAND+RERANK"]["recall"].append(recall_at_k(exp_rels, bullseye, good, K))
            all_metrics["EXPAND+RERANK"]["ap"].append(average_precision_at_k(exp_rels, K))
            all_metrics["EXPAND+RERANK"]["hit_rate"].append(hit_rate_at_k(exp_rels, K))

            print(f"\n  EXPAND+RERANK top-{K}:")
            for i, r in enumerate(exp_results[:K]):
                title = id_to_title.get(r["tmdb_id"], "?")
                rel = exp_rels[i]
                marker = " ★" if rel == 2 else " ●" if rel == 1 else ""
                print(f"    {i+1:>2}. {title[:45]}{marker}")

            time.sleep(0.3)

        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RETRIEVAL METRICS SUMMARY")
    print("=" * 90)
    print(f"\nAveraged over {len(TEST_QUERIES)} queries, K={K}")
    print(f"Relevance grading: ★ bullseye=2, ● good=1, miss=0\n")

    header = f"{'Metric':<16}"
    for name in approaches:
        header += f"  {name:<16}"
    print(header)
    print("─" * (16 + 18 * len(approaches)))

    metric_names = [
        ("NDCG@10", "ndcg"),
        ("MRR@10", "mrr"),
        ("P@10", "precision"),
        ("Recall@10", "recall"),
        ("MAP@10", "ap"),
        ("Hit Rate", "hit_rate"),
    ]

    summary = {}
    for display_name, key in metric_names:
        row = f"{display_name:<16}"
        for name in approaches:
            vals = all_metrics[name][key]
            avg = sum(vals) / len(vals) if vals else 0
            row += f"  {avg:<16.4f}"
            summary[f"{name}_{display_name}"] = avg
        print(row)

    # Best approach
    best_ndcg = max(approaches, key=lambda n: sum(all_metrics[n]["ndcg"]) / len(all_metrics[n]["ndcg"]))
    best_val = sum(all_metrics[best_ndcg]["ndcg"]) / len(all_metrics[best_ndcg]["ndcg"])
    base_val = sum(all_metrics["BASELINE"]["ndcg"]) / len(all_metrics["BASELINE"]["ndcg"])

    print(f"\nBest approach by NDCG@10: {best_ndcg} ({best_val:.4f})")
    if best_ndcg != "BASELINE" and base_val > 0:
        improvement = (best_val - base_val) / base_val * 100
        print(f"Improvement over baseline: +{improvement:.1f}%")

    # Save
    output_path = PROJECT_ROOT / "research" / "evaluation" / "ir_metrics.json"
    with open(output_path, "w") as f:
        json.dump({"per_query": {name: {k: v for k, v in metrics.items()}
                                 for name, metrics in all_metrics.items()},
                   "summary": summary}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main() 





