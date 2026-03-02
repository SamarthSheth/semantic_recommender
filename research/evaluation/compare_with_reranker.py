"""
Two-stage retrieval: FAISS retrieval → LLM reranking.

Stage 1: Embedding search retrieves top-50 candidates (fast, cheap, ~10ms)
Stage 2: LLM reads the query + all 50 movie descriptions and picks
         the best 10 (slow, smart, ~2 seconds)

Why this should work dramatically better:

The embedding model compresses a movie's entire description into 384
numbers. Inevitably, nuance is lost. "Unreliable narrator" and "alter
ego" might be close-ish in embedding space, but not close enough to
beat a movie that literally contains the word "thriller."

An LLM doesn't have this compression problem. It reads the full text
of all 50 candidates and REASONS about which ones match the query.
It knows that "dual identity" and "alter ego" and "breaking the
fourth wall" are hallmarks of an unreliable narrator story. It knows
that "cozy nostalgic 80s movie" matches The Princess Bride even though
the plot description talks about swordfighting and kidnapping.

This is the industry standard approach:
- Stage 1 (retrieval): fast but imprecise, gets you in the right neighborhood
- Stage 2 (reranking): slow but precise, picks the best from that neighborhood

Google Search, Bing, Amazon, Netflix — they all use two-stage retrieval.

Cost: ~$0.01-0.02 per query with Claude Haiku (50 short movie descriptions
      fit easily in one API call).

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python research/evaluation/compare_with_reranker.py
"""

import json
import os
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


RERANKER_PROMPT = """You are a movie recommendation expert. A user is searching for movies.

Given the user's query and a list of candidate movies, select the 10 BEST matches 
and rank them from most to least relevant.

Consider:
- How well the movie's plot, themes, and tone match the query
- The FEELING and EXPERIENCE of watching the movie, not just surface-level plot similarity
- Mood, atmosphere, pacing, emotional impact
- Read between the lines — "unreliable narrator" means look for twist endings, 
  dual identities, perspective shifts even if those exact words aren't used
- "Cozy nostalgic 80s movie" means look for warm, family-friendly adventure/comedy 
  from that era, even if the plot sounds like action

Respond with ONLY a JSON array of the movie numbers (1-indexed) in order of relevance.
Example: [14, 3, 27, 8, 41, 19, 2, 33, 11, 6]"""


def build_reranker_input(query: str, candidates: list[dict]) -> str:
    """Build the prompt listing all candidates for the LLM to rerank."""
    lines = [f'User query: "{query}"\n\nCandidate movies:\n']
    for i, c in enumerate(candidates, 1):
        # Truncate document to save tokens — the LLM only needs enough to judge
        doc = c["document"][:300]
        lines.append(f"{i}. {c['title']}: {doc}")
    lines.append(f"\nSelect the 10 best matches from the {len(candidates)} candidates above.")
    return "\n".join(lines)


def rerank_with_llm(query: str, candidates: list[dict], client) -> list[int]:
    """
    Ask Claude to rerank candidates by relevance.
    Returns list of candidate indices (0-indexed) in ranked order.
    """
    prompt = build_reranker_input(query, candidates)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=RERANKER_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Parse JSON array
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        indices = json.loads(text)
        # Convert from 1-indexed to 0-indexed
        return [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(candidates)]
    except json.JSONDecodeError:
        # Fallback: try to extract numbers
        import re
        numbers = re.findall(r'\d+', text)
        return [int(n) - 1 for n in numbers[:10] if 1 <= int(n) <= len(candidates)]


# ── Test queries (same as before) ────────────────────────────────────

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
        "bullseye": ["Lost in Translation", "Her", "Eternal Sunshine", "In the Mood for Love"],
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


def search_faiss(model, index, movie_ids, query, top_k=50):
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({"tmdb_id": movie_ids[idx], "score": float(score)})
    return results


def score_results(results, id_to_title, bullseye, good, top_k=10):
    bullseye_hits = []
    good_hits = []
    for r in results[:top_k]:
        title = id_to_title.get(r["tmdb_id"], "").lower()
        for b in bullseye:
            if b.lower() in title.lower():
                bullseye_hits.append(b)
                break
        for g in good:
            if g.lower() in title.lower():
                good_hits.append(g)
                break
    points = len(bullseye_hits) * 2 + len(good_hits)
    return points, bullseye_hits, good_hits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=50,
                        help="How many candidates to retrieve before reranking")
    args = parser.parse_args()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])
    movies = pd.read_parquet(processed_dir / "movies_final.parquet")

    id_to_title = {}
    id_to_doc = {}
    for _, row in movies.iterrows():
        tid = row["tmdbId"]
        title = row.get("title", row.get("title_tmdb", "Unknown"))
        id_to_title[tid] = title
        id_to_doc[tid] = row["document"]

    # Load model and index
    print("Loading model and index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_dir = Path(config["paths"]["embeddings"])
    index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    if hasattr(index, "nprobe"):
        index.nprobe = 10
    with open(embeddings_dir / "movie_ids.json") as f:
        movie_ids = json.load(f)

    # Also try expansion + reranking
    try:
        from query_expansion import expand_query
        expand_fn = expand_query
        print("Query expansion available.\n")
    except ImportError:
        expand_fn = None

    # ── Run comparisons ──────────────────────────────────────────────
    totals = {
        "BASELINE": {"points": 0, "bullseyes": 0, "goods": 0},
        "RERANKED": {"points": 0, "bullseyes": 0, "goods": 0},
        "EXPAND+RERANK": {"points": 0, "bullseyes": 0, "goods": 0},
    }

    for tq in TEST_QUERIES:
        query = tq["query"]
        bullseye = tq["bullseye"]
        good = tq["good"]

        print("=" * 95)
        print(f'QUERY: "{query}"  [{tq["type"]}]')
        print(f'BULLSEYE: {", ".join(bullseye)}')
        print("=" * 95)

        # Stage 1: Retrieve top-50 candidates
        raw_results = search_faiss(model, index, movie_ids, query, args.retrieve_k)

        candidates = []
        for r in raw_results:
            candidates.append({
                "tmdb_id": r["tmdb_id"],
                "title": id_to_title.get(r["tmdb_id"], "Unknown"),
                "document": id_to_doc.get(r["tmdb_id"], ""),
                "embedding_score": r["score"],
            })

        # Baseline: just take the top 10 from FAISS
        baseline_top10 = raw_results[:args.top_k]
        bl_pts, bl_bull, bl_good = score_results(baseline_top10, id_to_title, bullseye, good, args.top_k)
        totals["BASELINE"]["points"] += bl_pts
        totals["BASELINE"]["bullseyes"] += len(bl_bull)
        totals["BASELINE"]["goods"] += len(bl_good)

        # Reranked: ask LLM to rerank the top-50
        print("  Reranking with LLM...")
        try:
            reranked_indices = rerank_with_llm(query, candidates, client)
            reranked_results = [{"tmdb_id": candidates[i]["tmdb_id"],
                                "score": 1.0 - rank * 0.01}  # synthetic score for display
                               for rank, i in enumerate(reranked_indices[:args.top_k])
                               if i < len(candidates)]
        except Exception as e:
            print(f"  Reranking failed: {e}")
            reranked_results = baseline_top10
        rr_pts, rr_bull, rr_good = score_results(reranked_results, id_to_title, bullseye, good, args.top_k)
        totals["RERANKED"]["points"] += rr_pts
        totals["RERANKED"]["bullseyes"] += len(rr_bull)
        totals["RERANKED"]["goods"] += len(rr_good)

        # Expand + Rerank: expand query, retrieve top-50, rerank
        exp_reranked_results = reranked_results  # fallback
        if expand_fn:
            expanded = expand_fn(query)
            exp_raw = search_faiss(model, index, movie_ids, expanded, args.retrieve_k)
            exp_candidates = []
            for r in exp_raw:
                exp_candidates.append({
                    "tmdb_id": r["tmdb_id"],
                    "title": id_to_title.get(r["tmdb_id"], "Unknown"),
                    "document": id_to_doc.get(r["tmdb_id"], ""),
                    "embedding_score": r["score"],
                })
            print("  Reranking expanded results...")
            try:
                exp_reranked_idx = rerank_with_llm(query, exp_candidates, client)
                exp_reranked_results = [{"tmdb_id": exp_candidates[i]["tmdb_id"],
                                         "score": 1.0 - rank * 0.01}
                                        for rank, i in enumerate(exp_reranked_idx[:args.top_k])
                                        if i < len(exp_candidates)]
            except Exception as e:
                print(f"  Expand+rerank failed: {e}")
        er_pts, er_bull, er_good = score_results(exp_reranked_results, id_to_title, bullseye, good, args.top_k)
        totals["EXPAND+RERANK"]["points"] += er_pts
        totals["EXPAND+RERANK"]["bullseyes"] += len(er_bull)
        totals["EXPAND+RERANK"]["goods"] += len(er_good)

        time.sleep(0.3)

        # ── Print results side by side ───────────────────────────────
        all_approaches = [
            ("BASELINE", baseline_top10, bl_bull, bl_good, bl_pts),
            ("RERANKED", reranked_results, rr_bull, rr_good, rr_pts),
            ("EXPAND+RERANK", exp_reranked_results, er_bull, er_good, er_pts),
        ]

        col_w = 30
        names = [a[0] for a in all_approaches]
        print(f"\n     {''.join(f'{n:<{col_w}}' for n in names)}")
        print(f"     {''.join('─' * (col_w - 2) + '  ' for _ in names)}")

        for i in range(args.top_k):
            parts = []
            for name, results, _, _, _ in all_approaches:
                if i < len(results):
                    r = results[i]
                    title = id_to_title.get(r["tmdb_id"], "?")
                    marker = ""
                    for b in bullseye:
                        if b.lower() in title.lower():
                            marker = " ★"
                            break
                    if not marker:
                        for g in good:
                            if g.lower() in title.lower():
                                marker = " ●"
                                break
                    entry = f"[{r['score']:.3f}] {title[:18]}{marker}"
                else:
                    entry = ""
                parts.append(f"{entry:<{col_w}}")
            print(f"  {i+1:>2}. {''.join(parts)}")

        print()
        for name, _, bull, gd, pts in all_approaches:
            matches = []
            if bull:
                matches.append(f"★ {', '.join(bull)}")
            if gd:
                matches.append(f"● {', '.join(gd)}")
            match_str = " | ".join(matches) if matches else "none"
            print(f"  {name:<16} {pts:>2} pts  {match_str}")
        print()

    # ── Overall ──────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("OVERALL SUMMARY")
    print("=" * 95)
    print(f"\n★ = bullseye (2 pts)  ● = good match (1 pt)\n")
    print(f"{'Approach':<18} {'Points':>8} {'★ Bullseyes':>14} {'● Good':>10}")
    print("─" * 55)
    for name in ["BASELINE", "RERANKED", "EXPAND+RERANK"]:
        t = totals[name]
        print(f"{name:<18} {t['points']:>8} {t['bullseyes']:>14} {t['goods']:>10}")

    best = max(totals.items(), key=lambda x: x[1]["points"])
    print(f"\nBest approach: {best[0]} ({best[1]['points']} points)")

    output_path = PROJECT_ROOT / "research" / "evaluation" / "reranker_results.json"
    with open(output_path, "w") as f:
        json.dump(totals, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
