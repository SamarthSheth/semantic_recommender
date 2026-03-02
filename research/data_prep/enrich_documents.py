"""
Enrich movie documents with LLM-generated experiential descriptions.

== The Problem ==

Our movie documents describe PLOT: "A thief who commits corporate
espionage through dream-sharing technology."

User queries describe EXPERIENCE: "a mind-bending movie that makes
you question reality."

Query expansion helped the query side, but the documents still only
speak in plot language. We need the documents to ALSO contain
experiential language so there's vocabulary overlap.

== The Solution ==

For each movie, ask an LLM to generate a short "experience" description:
what it FEELS like to watch this movie, its mood, atmosphere, and the
emotional response it evokes. Then append this to the existing document.

Before:
  "Cobb, a skilled thief who commits corporate espionage by
   infiltrating the subconscious... Genre: Sci-Fi. Themes: dreams..."

After:
  "Cobb, a skilled thief who commits corporate espionage by
   infiltrating the subconscious... Genre: Sci-Fi. Themes: dreams...
   Experience: A mind-bending, cerebral thriller that layers reality
   upon reality until you question what's real. Builds mounting tension
   through impossible architecture and shifting dreamscapes. Leaves you
   thinking for days after watching."

Now when someone queries "mind-bending movie that makes you question
reality," there's direct vocabulary overlap with the enriched document.

== Cost ==

~12,000 movies × ~300 tokens per call = ~3.6M tokens
With Claude Haiku: ~$0.90 input + ~$4.50 output ≈ $5-6 total
Runtime: ~20-30 minutes

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python research/data_prep/enrich_documents.py --limit 20   # test first
    python research/data_prep/enrich_documents.py              # full run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from config_loader import load_config


SYSTEM_PROMPT = """You are a film critic helping build a movie recommendation system.

Given a movie's title, plot, genres, and themes, write a brief "experience" 
description: what it FEELS like to watch this movie.

Cover these aspects in 2-3 sentences (40-60 words total):
- Mood and atmosphere (tense, dreamy, cozy, bleak, euphoric)
- Pacing and style (slow burn, frenetic, meditative, propulsive)
- Emotional impact (heartbreaking, exhilarating, unsettling, uplifting)
- What kind of viewer or mood it's best for

Rules:
- Write in second person ("you feel", "leaves you")
- Be specific and evocative, not generic
- Don't summarize the plot — describe the EXPERIENCE
- Don't name other movies
- Keep it under 60 words

Respond with ONLY the experience description, no labels or preamble."""


def build_prompt(title, overview, genres, keywords, director):
    parts = [f"Movie: {title}"]
    if overview:
        parts.append(f"Plot: {overview[:300]}")
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if keywords:
        parts.append(f"Themes: {', '.join(keywords[:8])}")
    if director and str(director) != "nan":
        parts.append(f"Director: {director}")
    return "\n".join(parts)


def generate_experience(prompt, client):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])
    movies = pd.read_parquet(processed_dir / "movies_final.parquet")
    print(f"Total movies: {len(movies):,}")

    if args.limit:
        movies = movies.head(args.limit)
        print(f"Limited to: {len(movies):,}")

    # Checkpointing
    checkpoint_path = processed_dir / "enrichment_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            experiences = json.load(f)
        print(f"Resuming from checkpoint: {len(experiences)} already done")
    else:
        experiences = {}

    done_ids = set(experiences.keys())
    remaining = [(i, row) for i, row in movies.iterrows()
                 if str(int(row["tmdbId"])) not in done_ids]
    print(f"Remaining: {len(remaining):,}")

    errors = 0
    for idx, (i, row) in enumerate(tqdm(remaining, desc="Enriching")):
        tmdb_id = str(int(row["tmdbId"]))

        genres = json.loads(row["genres_tmdb"]) if isinstance(row["genres_tmdb"], str) else row["genres_tmdb"]
        keywords = json.loads(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]

        prompt = build_prompt(
            title=row.get("title", row.get("title_tmdb", "Unknown")),
            overview=row.get("overview", ""),
            genres=genres,
            keywords=keywords,
            director=row.get("director", ""),
        )

        try:
            experience = generate_experience(prompt, client)
            experiences[tmdb_id] = experience
        except Exception as e:
            errors += 1
            tqdm.write(f"  Error {tmdb_id}: {e}")
            if "rate" in str(e).lower():
                time.sleep(5)
            continue

        if (idx + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(experiences, f)
            tqdm.write(f"  Checkpoint: {len(experiences):,} done, {errors} errors")

        time.sleep(0.05)  # ~20 req/s

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(experiences, f)

    # ── Build enriched documents ─────────────────────────────────────
    print("\nBuilding enriched documents...")

    # Reload full movie set (in case we used --limit for generation)
    all_movies = pd.read_parquet(processed_dir / "movies_final.parquet")

    enriched_docs = []
    enriched_count = 0
    for _, row in all_movies.iterrows():
        tmdb_id = str(int(row["tmdbId"]))
        original_doc = row["document"]

        if tmdb_id in experiences:
            enriched_doc = f"{original_doc} Experience: {experiences[tmdb_id]}"
            enriched_count += 1
        else:
            enriched_doc = original_doc

        enriched_docs.append(enriched_doc)

    all_movies = all_movies.copy()
    all_movies["document_original"] = all_movies["document"]
    all_movies["document"] = enriched_docs

    # Save
    output_path = processed_dir / "movies_enriched.parquet"
    all_movies.to_parquet(output_path, index=False)

    print(f"\nEnriched {enriched_count:,} / {len(all_movies):,} movies")
    print(f"Saved to {output_path}")

    # Show samples
    print(f"\n{'=' * 70}")
    print("SAMPLE ENRICHED DOCUMENTS")
    print("=" * 70)

    samples = ["Fight Club", "Inception", "Lost in Translation",
               "Princess Bride", "Schindler"]
    for s in samples:
        match = all_movies[all_movies["title"].str.contains(s, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            tid = str(int(row["tmdbId"]))
            if tid in experiences:
                print(f"\n{row['title']}:")
                print(f"  Experience: {experiences[tid]}")


if __name__ == "__main__":
    main()
