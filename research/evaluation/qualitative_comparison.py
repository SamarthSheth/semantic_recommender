"""
Qualitative comparison of all three approaches.

The previous evaluation was too strict — it only counted exact matches
against a narrow list of "expected" movies. But recommendation is
subjective: if someone asks for "Blade Runner meets Her" and gets
The Matrix and eXistenZ, that's a GOOD result even though those
weren't on our expected list.

This script does two things:
1. Shows results side by side with clear formatting for eyeball evaluation
2. Uses a BROADER expected list with "tiers" — perfect matches and
   acceptable matches — for a more realistic metric

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python research/evaluation/qualitative_comparison.py
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend"))

from config_loader import load_config


# ── Expanded test queries with tiered expected results ───────────────
# Tier 1 ("bullseye"): the obvious correct answers anyone would agree on
# Tier 2 ("good"): reasonable recommendations that show the system understands the query

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


def search(model, index, movie_ids, query, top_k=20):
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({"tmdb_id": movie_ids[idx], "score": float(score)})
    return results


def score_results(results, id_to_title, bullseye, good, top_k=10):
    """
    Score results against tiered expected lists.
    Bullseye match = 2 points, good match = 1 point.
    Returns (points, max_possible, bullseye_hits, good_hits, matched_titles)
    """
    bullseye_hits = []
    good_hits = []

    for r in results[:top_k]:
        title = id_to_title.get(r["tmdb_id"], "").lower()
        for b in bullseye:
            if b.lower() in title:
                bullseye_hits.append(b)
                break
        for g in good:
            if g.lower() in title:
                good_hits.append(g)
                break

    points = len(bullseye_hits) * 2 + len(good_hits) * 1
    max_possible = min(len(bullseye), top_k) * 2  # theoretical max if all bullseyes found
    return points, max_possible, bullseye_hits, good_hits


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--no-expansion", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])
    movies = pd.read_parquet(processed_dir / "movies_final.parquet")

    id_to_title = {}
    for _, row in movies.iterrows():
        id_to_title[row["tmdbId"]] = row.get("title", row.get("title_tmdb", "Unknown"))

    # Load baseline
    print("Loading baseline model...")
    baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_dir = Path(config["paths"]["embeddings"])
    baseline_index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    if hasattr(baseline_index, "nprobe"):
        baseline_index.nprobe = 10
    with open(embeddings_dir / "movie_ids.json") as f:
        baseline_ids = json.load(f)

    # Load fine-tuned
    ft_model = None
    ft_index = None
    ft_ids = None
    ft_path = PROJECT_ROOT / "models" / "baseline" / "model"
    if ft_path.exists():
        print("Loading fine-tuned model...")
        ft_model = SentenceTransformer(str(ft_path))
        print("Encoding movies with fine-tuned model...")
        docs = movies["document"].tolist()
        ft_ids = movies["tmdbId"].tolist()
        ft_emb = ft_model.encode(docs, batch_size=128, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        ft_index = faiss.IndexFlatIP(ft_emb.shape[1])
        ft_index.add(ft_emb)

    # Setup expansion
    expand_fn = None
    if not args.no_expansion:
        try:
            from query_expansion import expand_query
            if os.environ.get("ANTHROPIC_API_KEY"):
                expand_fn = expand_query
                print("Query expansion enabled.\n")
        except ImportError:
            pass

    # Run
    totals = {
        "baseline": {"points": 0, "bullseyes": 0, "goods": 0},
        "fine_tuned": {"points": 0, "bullseyes": 0, "goods": 0},
        "expanded": {"points": 0, "bullseyes": 0, "goods": 0},
    }

    for tq in TEST_QUERIES:
        query = tq["query"]
        bullseye = tq["bullseye"]
        good = tq["good"]

        print("=" * 90)
        print(f'QUERY: "{query}"  [{tq["type"]}]')
        print(f'BULLSEYE: {", ".join(bullseye)}')
        print("=" * 90)

        approaches = {}

        # Baseline
        bl_results = search(baseline_model, baseline_index, baseline_ids, query, args.top_k)
        bl_pts, _, bl_bull, bl_good = score_results(bl_results, id_to_title, bullseye, good, args.top_k)
        approaches["baseline"] = (bl_results, bl_bull, bl_good, bl_pts)
        totals["baseline"]["points"] += bl_pts
        totals["baseline"]["bullseyes"] += len(bl_bull)
        totals["baseline"]["goods"] += len(bl_good)

        # Fine-tuned
        if ft_model:
            ft_results = search(ft_model, ft_index, ft_ids, query, args.top_k)
            ft_pts, _, ft_bull, ft_good = score_results(ft_results, id_to_title, bullseye, good, args.top_k)
            approaches["fine_tuned"] = (ft_results, ft_bull, ft_good, ft_pts)
            totals["fine_tuned"]["points"] += ft_pts
            totals["fine_tuned"]["bullseyes"] += len(ft_bull)
            totals["fine_tuned"]["goods"] += len(ft_good)

        # Expanded
        if expand_fn:
            expanded_text = expand_fn(query)
            print(f'\nEXPANDED → "{expanded_text[:200]}..."')
            exp_results = search(baseline_model, baseline_index, baseline_ids, expanded_text, args.top_k)
            exp_pts, _, exp_bull, exp_good = score_results(exp_results, id_to_title, bullseye, good, args.top_k)
            approaches["expanded"] = (exp_results, exp_bull, exp_good, exp_pts)
            totals["expanded"]["points"] += exp_pts
            totals["expanded"]["bullseyes"] += len(exp_bull)
            totals["expanded"]["goods"] += len(exp_good)
            time.sleep(0.3)

        # Print results
        approach_names = list(approaches.keys())
        col_w = 28

        print(f"\n     {''.join(f'{n.upper():<{col_w}}' for n in approach_names)}")
        print(f"     {''.join('─' * (col_w - 2) + '  ' for _ in approach_names)}")

        for i in range(args.top_k):
            parts = []
            for name in approach_names:
                results, _, _, _ = approaches[name]
                if i < len(results):
                    r = results[i]
                    title = id_to_title.get(r["tmdb_id"], "?")

                    # Mark matches
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

        # Per-query summary
        print()
        for name in approach_names:
            _, bull, gd, pts = approaches[name]
            bull_str = ", ".join(bull) if bull else "none"
            good_str = ", ".join(gd) if gd else "none"
            print(f"  {name.upper():<14} ★ bullseye: {bull_str}")
            if gd:
                print(f"  {'':14} ● good: {good_str}")
            print(f"  {'':14} Score: {pts} pts")
        print()

    # ── Overall summary ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)
    print(f"\n★ = bullseye (2 pts)  ● = good match (1 pt)")
    print(f"\n{'Approach':<16} {'Points':>8} {'★ Bullseyes':>14} {'● Good':>10}")
    print("─" * 52)
    for name in ["baseline", "fine_tuned", "expanded"]:
        t = totals[name]
        if t["points"] > 0 or name == "baseline":
            print(f"{name:<16} {t['points']:>8} {t['bullseyes']:>14} {t['goods']:>10}")

    # Winner
    best = max(totals.items(), key=lambda x: x[1]["points"])
    print(f"\nBest approach: {best[0]} ({best[1]['points']} points)")


if __name__ == "__main__":
    main()
