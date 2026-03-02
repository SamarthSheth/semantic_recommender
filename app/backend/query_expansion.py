"""
LLM-powered query expansion for semantic movie retrieval.

Instead of retraining the embedding model, we transform the user's
vibes-based query into descriptive language that the frozen embeddings
already understand.

User: "something that feels like a rainy Sunday afternoon"
  ↓ LLM expands
"A quiet, contemplative drama with melancholic tone, themes of
solitude and introspection, slow pacing, muted cinematography.
Similar to films about characters reflecting on life, featuring
atmospheric settings and understated emotional depth."
  ↓ Embed expanded text
Search FAISS → much better results

Why this works:
The frozen sentence-transformer already understands that "melancholic"
is close to "sad drama" and "slow pacing" is close to "contemplative."
The problem was never the embedding model — it was that "rainy Sunday
afternoon" doesn't contain any of those words. The LLM bridges that
gap at query time, translating vibes into vocabulary the embeddings
know how to handle.

Trade-offs vs fine-tuning:
+ No training required, works immediately
+ Can improve with better prompts (no retraining)
+ Handles novel query types the training data never covered
- Adds ~1-2 seconds latency per query (LLM API call)
- Costs money per query (small with Haiku, but nonzero)
- Dependent on external API availability
"""

import os
import re

import anthropic


EXPANSION_PROMPT = """You are helping a movie recommendation system understand what the user wants.

The user typed a search query. Your job is to expand it into a detailed description that would match movie plot summaries and metadata. 

Transform vibes, moods, and abstract feelings into concrete film attributes:
- Genres and subgenres
- Tone and atmosphere
- Pacing and style
- Themes and subject matter
- Types of characters or settings
- Similar directors or film movements (without naming specific movies)

Rules:
- Write 2-3 sentences, roughly 50-80 words
- Use descriptive language that would appear in a movie's plot summary or review
- Don't name specific movies or characters
- Don't repeat the user's query verbatim
- Focus on translating abstract feelings into concrete film attributes

Examples:
User: "something that feels like a rainy Sunday afternoon"
Expansion: A quiet, contemplative drama with a melancholic but gentle tone. Slow pacing with atmospheric cinematography, exploring themes of solitude, reflection, and finding peace in stillness. Characters navigating personal crossroads or quiet moments of connection, set against intimate or urban backdrops.

User: "a movie that will make me ugly cry"
Expansion: An emotionally devastating drama with powerful performances exploring themes of loss, sacrifice, love, or family bonds. Stories about terminal illness, grief, separation, or unexpected kindness in dark times. Films that build emotional investment before delivering a deeply moving climax.

User: "weird surreal film that messes with your head"
Expansion: A mind-bending psychological film that blurs reality and fantasy, featuring unreliable narrators, dream sequences, or fractured timelines. Surrealist or avant-garde visual style with unsettling atmosphere. Themes of identity, perception, paranoia, or existential questioning that challenge the viewer's understanding."""


def expand_query(query: str, api_key: str = None) -> str:
    """
    Expand a user's vibes-based query into descriptive movie language.

    Returns the expanded query text, or the original query if
    expansion fails (graceful degradation).
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # No API key — fall back to original query
        return query

    try:
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=EXPANSION_PROMPT,
            messages=[{
                "role": "user",
                "content": f"User query: \"{query}\"\n\nExpansion:"
            }],
        )

        expanded = response.content[0].text.strip()

        # Combine original query with expansion for best coverage
        # The original captures any literal terms the user mentioned
        # The expansion adds the descriptive vocabulary
        combined = f"{query}. {expanded}"
        return combined

    except Exception as e:
        print(f"Query expansion failed: {e}")
        return query


def expand_query_batch(queries: list[str], api_key: str = None) -> list[str]:
    """Expand multiple queries. Used for batch evaluation."""
    return [expand_query(q, api_key) for q in queries]
