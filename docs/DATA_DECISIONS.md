# Data Source Decisions

## Why TMDB as the primary movie catalog

We need rich text per movie to build meaningful embeddings. TMDB provides:
- Plot overviews (1-3 paragraph summaries)
- Genre labels
- Keywords/tags (thematic descriptors like "revenge", "dystopia")
- Cast and crew
- User reviews (with ratings)
- Release dates, budgets, popularity scores

Alternative considered: **OMDb/IMDb** — richer plot summaries but API is more
restrictive and harder to bulk-download.

Alternative considered: **Wikipedia plot summaries** — very detailed but
inconsistent formatting and length. Could supplement TMDB later.

## Why MovieLens as the ratings source

MovieLens 25M gives us 25 million ratings from 162,000 users across 62,000 movies.
We use this for:
1. Filtering: only keep movies that real users have actually watched and rated
2. Future collaborative filtering signal if we build a hybrid model
3. Evaluation: we can hold out ratings to test if our semantic model
   recommends movies users actually liked

MovieLens movies link to TMDB via `links.csv` (contains tmdbId).
This is the join key between our two sources.

## Why NOT Amazon Reviews as the primary text source

Amazon Movie Reviews have rich text but:
- Linking Amazon products to TMDB movies is fuzzy string matching (error-prone)
- Many are DVD/Blu-ray reviews, not movie reviews ("shipped fast, great packaging")
- TMDB reviews are already linked to the right movie by ID

## Data flow

MovieLens (ratings + tmdbIds) → filter to well-rated movies
                                      ↓
                              TMDB API (metadata + reviews per tmdbId)
                                      ↓
                              Merged dataset in parquet format
