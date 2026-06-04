"""
CineMatch AI — Synopsis Embeddings
Generates 384-dim multilingual sentence embeddings for catalog overviews.
Saves to data/embeddings.npy (one row per catalog row, same order as catalog.parquet).

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 120 MB download (cached after first run)
  - Handles Hebrew + English queries natively
  - 384-dim output
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT_PATH   = os.path.join(BASE, "data", "catalog.parquet")
OUT_PATH   = os.path.join(BASE, "data", "embeddings.npy")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256


def main():
    print(f"Loading catalog from {CAT_PATH}...")
    df = pd.read_parquet(CAT_PATH)
    print(f"  {len(df)} rows")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Use overview if available, else fall back to "title genres"
    texts = []
    for _, row in df.iterrows():
        ov = str(row.get("overview", "") or "").strip()
        if ov and len(ov) > 10:
            texts.append(ov)
        else:
            texts.append(f"{row['title']} {row.get('genres','')}")

    print(f"Encoding {len(texts)} texts in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so dot product = cosine similarity
    )

    np.save(OUT_PATH, embeddings.astype(np.float32))
    print(f"Saved embeddings → {OUT_PATH}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
