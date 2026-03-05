"""
Compare three retrieval approaches side-by-side:

1. BASELINE:         Frozen embeddings, raw query
2. FINE-TUNED:       Contrastive fine-tuned embeddings, raw query
3. QUERY EXPANSION:  Frozen embeddings, LLM-expanded query


Usage:
    export ANTHROPIC_API_KEY="your-key"
    cd ~/Downloads/semantic-rec
    python research/evaluation/compare_all_approaches.py

    # Without query expansion (skips LLM calls):
    python research/evaluation/compare_all_approaches.py --no-expansion

Output:
    - Side-by-side results printed to terminal
    - research/evaluation/comparison_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend"))

from config_loader import load_config


# ── Test Queries ─────────────────────────────────────────────────────
# These are designed to test different query types that a real user
# would type. Each has "expected" movies — not exhaustive ground truth,
# just movies that SHOULD appear if the system is working well.
# This is our curated benchmark.

TEST_QUERIES = [
    {
        "query": "a dark psychological thriller with an unreliable narrator",
        "type": "genre+concept",
        "expected": ["Fight Club", "Shutter Island", "Black Swan",
                     "Gone Girl", "Memento", "Sixth Sense"],
    },
    {
        "query": "feel-good comedy about friendship and growing up",
        "type": "mood+theme",
        "expected": ["Superbad", "Stand by Me", "The Breakfast Club",
                     "Ferris Bueller", "Dazed and Confused"],
    },
    {
        "query": "visually stunning sci-fi epic about space exploration",
        "type": "style+genre",
        "expected": ["Interstellar", "2001", "Gravity", "Arrival",
                     "The Martian", "Ad Astra"],
    },
    {
        "query": "quiet, melancholic drama set in a small town",
        "type": "mood+setting",
        "expected": ["Manchester by the Sea", "Nebraska", "Nomadland",
                     "Three Billboards", "Winter's Bone"],
    },
    {
        "query": "fast-paced heist movie with clever twists",
        "type": "pace+genre",
        "expected": ["Ocean's Eleven", "The Italian Job", "Inside Man",
                     "Heat", "Baby Driver", "Now You See Me"],
    },
    {
        "query": "something that feels like a rainy Sunday afternoon",
        "type": "vibes",
        "expected": ["Lost in Translation", "Her", "Eternal Sunshine",
                     "Amelie", "In the Mood for Love"],
    },
    {
        "query": "a movie that will make me ugly cry",
        "type": "emotional",
        "expected": ["Schindler's List", "The Green Mile", "Up",
                     "Grave of the Fireflies", "Life Is Beautiful"],
    },
    {
        "query": "weird surreal film that messes with your head",
        "type": "style+feeling",
        "expected": ["Mulholland Drive", "Eraserhead", "Donnie Darko",
                     "Being John Malkovich", "Synecdoche, New York"],
    },
    {
        "query": "cozy nostalgic movie from the 80s or 90s",
        "type": "mood+era",
        "expected": ["The Princess Bride", "Back to the Future",
                     "Home Alone", "E.T.", "The Goonies"],
    },
    {
        "query": "intense courtroom drama with incredible acting",
        "type": "genre+quality",
        "expected": ["12 Angry Men", "A Few Good Men", "To Kill a Mockingbird",
                     "Primal Fear", "The Verdict"],
    },
    {
        "query": "like Blade Runner meets Her",
        "type": "comparative",
        "expected": ["Ex Machina", "A.I.", "Ghost in the Shell",
                     "Solaris", "Eternal Sunshine"],
    },
    {
        "query": "that anxious feeling when everything is about to fall apart",
        "type": "abstract_emotion",
        "expected": ["Uncut Gems", "Requiem for a Dream", "Black Swan",
                     "Whiplash", "Climax"],
    },
    {
        "query": "beautiful foreign film about family",
        "type": "quality+theme",
        "expected": ["Shoplifters", "Roma", "Parasite", "Amour",
                     "The Farewell", "Like Father, Like Son"],
    },
    {
        "query": "a smart movie that respects the audience's intelligence",
        "type": "meta_quality",
        "expected": ["Primer", "Arrival", "The Prestige", "Inception",
                     "Zodiac", "Tinker Tailor Soldier Spy"],
    },
    {
        "query": "something dark and atmospheric with amazing cinematography",
        "type": "style+mood",
        "expected": ["Blade Runner 2049", "The Revenant", "Sicario",
                     "No Country for Old Men", "There Will Be Blood"],
    },
]


# ── Search Functions ─────────────────────────────────────────────────

def search(model, index, movie_ids, query, top_k=20):
    """Encode query and search FAISS index."""
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "tmdb_id": movie_ids[idx],
                "score": float(score),
                "rank": len(results) + 1,
            })
    return results


def check_expected(results, expected_titles, id_to_title, top_k=10):
    """
    Check how many expected movies appear in the top-K results.
    Uses fuzzy substring matching since titles may differ slightly
    between our expected list and the actual dataset.
    """
    result_titles = []
    for r in results[:top_k]:
        title = id_to_title.get(r["tmdb_id"], "")
        result_titles.append(title.lower())

    hits = []
    for expected in expected_titles:
        expected_lower = expected.lower()
        found = any(expected_lower in rt for rt in result_titles)
        hits.append(found)

    return hits


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-expansion", action="store_true",
                        help="Skip query expansion (no LLM API calls)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to show per query")
    args = parser.parse_args()

    config = load_config()

    # ── Load movie metadata ──────────────────────────────────────────
    processed_dir = Path(config["paths"]["processed_data"])
    movies = pd.read_parquet(processed_dir / "movies_final.parquet")

    id_to_title = {}
    id_to_info = {}
    for _, row in movies.iterrows():
        tid = row["tmdbId"]
        title = row.get("title", row.get("title_tmdb", "Unknown"))
        genres = json.loads(row["genres_tmdb"]) if isinstance(row["genres_tmdb"], str) else row["genres_tmdb"]
        id_to_title[tid] = title
        id_to_info[tid] = {"title": title, "genres": genres}

    # ── Load baseline ────────────────────────────────────────────────
    print("Loading baseline model...")
    baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_dir = Path(config["paths"]["embeddings"])
    baseline_index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    if hasattr(baseline_index, "nprobe"):
        baseline_index.nprobe = 10
    with open(embeddings_dir / "movie_ids.json") as f:
        baseline_movie_ids = json.load(f)

    # ── Load fine-tuned ──────────────────────────────────────────────
    finetuned_model = None
    ft_index = None
    ft_movie_ids = None

    ft_model_path = PROJECT_ROOT / "models" / "baseline" / "model"
    if ft_model_path.exists():
        print("Loading fine-tuned model...")
        finetuned_model = SentenceTransformer(str(ft_model_path))

        print("Encoding movies with fine-tuned model...")
        documents = movies["document"].tolist()
        ft_movie_ids = movies["tmdbId"].tolist()
        ft_embeddings = finetuned_model.encode(
            documents, batch_size=128, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        ft_index = faiss.IndexFlatIP(ft_embeddings.shape[1])
        ft_index.add(ft_embeddings)
    else:
        print("No fine-tuned model found, skipping.")

    # ── Setup query expansion ────────────────────────────────────────
    expand_fn = None
    if not args.no_expansion:
        try:
            from query_expansion import expand_query
            import os
            if os.environ.get("ANTHROPIC_API_KEY"):
                expand_fn = expand_query
                print("Query expansion enabled.")
            else:
                print("ANTHROPIC_API_KEY not set, skipping query expansion.")
        except ImportError:
            print("Could not import query_expansion, skipping.")

    # ── Run comparison ───────────────────────────────────────────────
    print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")

    all_results = []
    approach_scores = {
        "baseline": {"hits": 0, "total_expected": 0, "queries_with_hit": 0},
        "fine_tuned": {"hits": 0, "total_expected": 0, "queries_with_hit": 0},
        "expanded": {"hits": 0, "total_expected": 0, "queries_with_hit": 0},
    }

    for tq in TEST_QUERIES:
        query = tq["query"]
        query_type = tq["type"]
        expected = tq["expected"]

        print("=" * 90)
        print(f'QUERY: "{query}"  [{query_type}]')
        print(f'EXPECTED: {", ".join(expected)}')
        print("=" * 90)

        query_result = {
            "query": query,
            "type": query_type,
            "expected": expected,
            "approaches": {},
        }

        # ── Baseline ────────────────────────────────────────────────
        baseline_results = search(
            baseline_model, baseline_index, baseline_movie_ids, query, args.top_k
        )
        baseline_hits = check_expected(baseline_results, expected, id_to_title, args.top_k)
        baseline_found = [e for e, h in zip(expected, baseline_hits) if h]
        approach_scores["baseline"]["hits"] += sum(baseline_hits)
        approach_scores["baseline"]["total_expected"] += len(expected)
        if any(baseline_hits):
            approach_scores["baseline"]["queries_with_hit"] += 1

        query_result["approaches"]["baseline"] = {
            "results": [(id_to_info.get(r["tmdb_id"], {}).get("title", "?"), r["score"])
                       for r in baseline_results[:args.top_k]],
            "expected_found": baseline_found,
        }

        # ── Fine-tuned ──────────────────────────────────────────────
        ft_found = []
        if finetuned_model:
            ft_results = search(
                finetuned_model, ft_index, ft_movie_ids, query, args.top_k
            )
            ft_hits = check_expected(ft_results, expected, id_to_title, args.top_k)
            ft_found = [e for e, h in zip(expected, ft_hits) if h]
            approach_scores["fine_tuned"]["hits"] += sum(ft_hits)
            approach_scores["fine_tuned"]["total_expected"] += len(expected)
            if any(ft_hits):
                approach_scores["fine_tuned"]["queries_with_hit"] += 1

            query_result["approaches"]["fine_tuned"] = {
                "results": [(id_to_info.get(r["tmdb_id"], {}).get("title", "?"), r["score"])
                           for r in ft_results[:args.top_k]],
                "expected_found": ft_found,
            }

        # ── Query expansion ─────────────────────────────────────────
        expanded_found = []
        expanded_text = None
        if expand_fn:
            expanded_text = expand_fn(query)
            exp_results = search(
                baseline_model, baseline_index, baseline_movie_ids,
                expanded_text, args.top_k
            )
            exp_hits = check_expected(exp_results, expected, id_to_title, args.top_k)
            expanded_found = [e for e, h in zip(expected, exp_hits) if h]
            approach_scores["expanded"]["hits"] += sum(exp_hits)
            approach_scores["expanded"]["total_expected"] += len(expected)
            if any(exp_hits):
                approach_scores["expanded"]["queries_with_hit"] += 1

            query_result["approaches"]["expanded"] = {
                "expanded_query": expanded_text,
                "results": [(id_to_info.get(r["tmdb_id"], {}).get("title", "?"), r["score"])
                           for r in exp_results[:args.top_k]],
                "expected_found": expanded_found,
            }

            # Small delay to avoid rate limits
            time.sleep(0.5)

        # ── Print side by side ──────────────────────────────────────
        if expanded_text:
            print(f'\nEXPANDED: "{expanded_text[:150]}..."')

        # Determine which columns to show
        headers = ["BASELINE"]
        result_cols = [baseline_results]
        found_cols = [baseline_found]

        if finetuned_model:
            headers.append("FINE-TUNED")
            result_cols.append(ft_results)
            found_cols.append(ft_found)

        if expand_fn:
            headers.append("EXPANDED")
            result_cols.append(exp_results)
            found_cols.append(expanded_found)

        # Print header
        col_width = 30
        header_str = "     " + "".join(f"{h:<{col_width}}" for h in headers)
        print(f"\n{header_str}")
        print("     " + "".join("-" * (col_width - 2) + "  " for _ in headers))

        # Print results
        for i in range(min(args.top_k, 10)):
            row_parts = []
            for col_results in result_cols:
                if i < len(col_results):
                    r = col_results[i]
                    title = id_to_info.get(r["tmdb_id"], {}).get("title", "?")
                    # Mark expected movies with ✓
                    marker = ""
                    for exp in expected:
                        if exp.lower() in title.lower():
                            marker = " ✓"
                            break
                    entry = f"[{r['score']:.3f}] {title[:20]}{marker}"
                else:
                    entry = ""
                row_parts.append(f"{entry:<{col_width}}")
            print(f"  {i+1:>2}. {''.join(row_parts)}")

        # Print hit summary for this query
        hit_parts = []
        for header, found in zip(headers, found_cols):
            hit_parts.append(f"{header}: {len(found)}/{len(expected)}")
        print(f"\n  Expected found → {' | '.join(hit_parts)}")
        print()

        all_results.append(query_result)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    print(f"\n{'Approach':<20} {'Expected Found':>16} {'Hit Rate':>10} {'Queries w/ ≥1 Hit':>20}")
    print("-" * 70)

    for approach, scores in approach_scores.items():
        if scores["total_expected"] == 0:
            continue
        hit_rate = scores["hits"] / scores["total_expected"]
        query_hit_rate = scores["queries_with_hit"] / len(TEST_QUERIES)
        print(
            f"{approach:<20} "
            f"{scores['hits']:>6}/{scores['total_expected']:<9} "
            f"{hit_rate:>9.1%} "
            f"{scores['queries_with_hit']:>10}/{len(TEST_QUERIES)}"
        )

    # ── Save detailed results ────────────────────────────────────────
    output_path = PROJECT_ROOT / "research" / "evaluation" / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "test_queries": all_results,
            "summary": approach_scores,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
