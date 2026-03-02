"""
Rebuild FAISS index from enriched documents and compare all approaches.

After running enrich_documents.py, this script:
1. Loads the enriched movie documents
2. Re-embeds them with the baseline model
3. Runs the same test queries against the enriched index
4. Compares: baseline_original vs baseline_enriched vs enriched+expansion

This is the key experiment. If enrichment works, we should see a big
jump because BOTH sides now speak experiential language:
- Query: "mind-bending movie" (or LLM-expanded version)
- Document: "...Experience: A cerebral thriller that layers reality
  upon reality until you question what's real..."

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python research/evaluation/compare_enriched.py
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


# Same test queries as qualitative_comparison.py
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


def search(model, index, movie_ids, query, top_k=10):
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
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--no-expansion", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])

    # ── Load enriched movies ─────────────────────────────────────────
    enriched_path = processed_dir / "movies_enriched.parquet"
    if not enriched_path.exists():
        print("ERROR: Run enrich_documents.py first!")
        sys.exit(1)

    movies_enriched = pd.read_parquet(enriched_path)
    movies_original = movies_enriched.copy()
    movies_original["document"] = movies_original["document_original"]

    id_to_title = {}
    for _, row in movies_enriched.iterrows():
        id_to_title[row["tmdbId"]] = row.get("title", row.get("title_tmdb", "Unknown"))

    # ── Load baseline model ──────────────────────────────────────────
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Build original index (for comparison) ────────────────────────
    print("Building original document index...")
    orig_docs = movies_original["document"].tolist()
    orig_ids = movies_original["tmdbId"].tolist()
    orig_emb = model.encode(orig_docs, batch_size=128, show_progress_bar=True,
                             normalize_embeddings=True).astype(np.float32)
    orig_index = faiss.IndexFlatIP(orig_emb.shape[1])
    orig_index.add(orig_emb)

    # ── Build enriched index ─────────────────────────────────────────
    print("Building enriched document index...")
    enr_docs = movies_enriched["document"].tolist()
    enr_ids = movies_enriched["tmdbId"].tolist()
    enr_emb = model.encode(enr_docs, batch_size=128, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)
    enr_index = faiss.IndexFlatIP(enr_emb.shape[1])
    enr_index.add(enr_emb)

    # ── Setup expansion ──────────────────────────────────────────────
    expand_fn = None
    if not args.no_expansion:
        try:
            from query_expansion import expand_query
            if os.environ.get("ANTHROPIC_API_KEY"):
                expand_fn = expand_query
                print("Query expansion enabled.\n")
        except ImportError:
            pass

    # ── Run comparison ───────────────────────────────────────────────
    # Four approaches:
    # A) Original docs, raw query (old baseline)
    # B) Original docs, expanded query (what we just tested)
    # C) Enriched docs, raw query (new — does enrichment alone help?)
    # D) Enriched docs, expanded query (the full system)

    approaches_config = [
        ("ORIGINAL", orig_index, orig_ids, False),
        ("ORIG+EXPAND", orig_index, orig_ids, True),
        ("ENRICHED", enr_index, enr_ids, False),
        ("ENR+EXPAND", enr_index, enr_ids, True),
    ]

    # Filter out expansion approaches if no API key
    if not expand_fn:
        approaches_config = [a for a in approaches_config if not a[3]]

    totals = {name: {"points": 0, "bullseyes": 0, "goods": 0}
              for name, _, _, _ in approaches_config}

    for tq in TEST_QUERIES:
        query = tq["query"]
        bullseye = tq["bullseye"]
        good = tq["good"]

        print("=" * 100)
        print(f'QUERY: "{query}"  [{tq["type"]}]')
        print(f'BULLSEYE: {", ".join(bullseye)}')
        print("=" * 100)

        # Get expanded query once
        expanded_text = None
        if expand_fn:
            expanded_text = expand_fn(query)
            print(f'EXPANDED → "{expanded_text[:150]}..."')
            time.sleep(0.3)

        results = {}
        for name, index, ids, use_expansion in approaches_config:
            q = expanded_text if (use_expansion and expanded_text) else query
            res = search(model, index, ids, q, args.top_k)
            pts, bull, gd = score_results(res, id_to_title, bullseye, good, args.top_k)
            results[name] = (res, bull, gd, pts)
            totals[name]["points"] += pts
            totals[name]["bullseyes"] += len(bull)
            totals[name]["goods"] += len(gd)

        # Print
        names = list(results.keys())
        col_w = 25
        print(f"\n     {''.join(f'{n:<{col_w}}' for n in names)}")
        print(f"     {''.join('─' * (col_w - 2) + '  ' for _ in names)}")

        for i in range(args.top_k):
            parts = []
            for name in names:
                res_list, _, _, _ = results[name]
                if i < len(res_list):
                    r = res_list[i]
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
                    entry = f"[{r['score']:.3f}] {title[:16]}{marker}"
                else:
                    entry = ""
                parts.append(f"{entry:<{col_w}}")
            print(f"  {i+1:>2}. {''.join(parts)}")

        # Per-query scores
        print()
        for name in names:
            _, bull, gd, pts = results[name]
            parts = []
            if bull:
                parts.append(f"★ {', '.join(bull)}")
            if gd:
                parts.append(f"● {', '.join(gd)}")
            match_str = " | ".join(parts) if parts else "none"
            print(f"  {name:<14} {pts:>2} pts  {match_str}")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"\n★ = bullseye (2 pts)  ● = good match (1 pt)\n")
    print(f"{'Approach':<16} {'Points':>8} {'★ Bullseyes':>14} {'● Good':>10} {'Description'}")
    print("─" * 85)

    descriptions = {
        "ORIGINAL": "Baseline (plot-only docs, raw query)",
        "ORIG+EXPAND": "Plot-only docs + LLM query expansion",
        "ENRICHED": "Enriched docs (plot+vibes), raw query",
        "ENR+EXPAND": "Enriched docs + LLM query expansion",
    }

    for name in [n for n, _, _, _ in approaches_config]:
        t = totals[name]
        desc = descriptions.get(name, "")
        print(f"{name:<16} {t['points']:>8} {t['bullseyes']:>14} {t['goods']:>10}  {desc}")

    best = max(totals.items(), key=lambda x: x[1]["points"])
    print(f"\nBest approach: {best[0]} ({best[1]['points']} points)")

    # Save
    output_path = PROJECT_ROOT / "research" / "evaluation" / "enriched_comparison.json"
    with open(output_path, "w") as f:
        json.dump(totals, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
