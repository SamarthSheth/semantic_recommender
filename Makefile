# Pipeline orchestration.
# Each target depends on the previous step's output file,
# so `make embeddings` will automatically run all prior steps if needed.

PYTHON = python
SCRIPTS = scripts
PROCESSED = data/processed
EMBEDDINGS = data/embeddings

# ── Pipeline steps ──────────────────────────────────────────────────

$(PROCESSED)/movielens_filtered.parquet:
	cd $(SCRIPTS) && $(PYTHON) 01_download_movielens.py

$(PROCESSED)/tmdb_metadata.parquet: $(PROCESSED)/movielens_filtered.parquet
	cd $(SCRIPTS) && $(PYTHON) 02_fetch_tmdb.py

$(PROCESSED)/movies_final.parquet: $(PROCESSED)/tmdb_metadata.parquet
	cd $(SCRIPTS) && $(PYTHON) 03_merge_and_build_documents.py

$(EMBEDDINGS)/faiss_index.bin: $(PROCESSED)/movies_final.parquet
	cd $(SCRIPTS) && $(PYTHON) 04_build_embeddings.py

# ── Convenience targets ─────────────────────────────────────────────

.PHONY: movielens tmdb data embeddings serve clean

movielens: $(PROCESSED)/movielens_filtered.parquet
tmdb: $(PROCESSED)/tmdb_metadata.parquet
data: $(PROCESSED)/movies_final.parquet
embeddings: $(EMBEDDINGS)/faiss_index.bin

# Run the full pipeline (download → process → embed)
all: embeddings

# Start the API server
serve: embeddings
	uvicorn app.backend.main:app --reload --port 8000

# Wipe all generated data (keeps raw downloads)
clean:
	rm -f $(PROCESSED)/*.parquet
	rm -f $(EMBEDDINGS)/*
	@echo "Cleaned processed data and embeddings. Raw downloads preserved."

# Nuclear option — wipe everything including downloads
clean-all: clean
	rm -rf data/raw/*
	@echo "Cleaned everything."
