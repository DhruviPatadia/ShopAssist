from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_policies(policy_dir):
    """Load all Markdown policy files from a directory."""
    policy_dir = Path(policy_dir)
    documents = []

    for file_path in sorted(policy_dir.glob("*.md")):
        documents.append({
            "source": file_path.name,
            "text": file_path.read_text(encoding="utf-8")
        })

    if not documents:
        raise ValueError(f"No Markdown policy files found in {policy_dir}")

    return pd.DataFrame(documents)


def build_retrieval_index(documents_df, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create normalized embeddings and a FAISS similarity-search index."""
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        documents_df["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def search_policies(query, documents_df, model, index, top_k=3):
    """Return the top matching policy documents and similarity scores."""
    if not query or not query.strip():
        return pd.DataFrame(columns=["rank", "source", "similarity"])

    query_embedding = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(documents_df))
    scores, indices = index.search(query_embedding, k)

    results = []
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx >= 0:
            results.append({
                "rank": rank,
                "source": documents_df.iloc[idx]["source"],
                "similarity": float(score)
            })

    return pd.DataFrame(results)
