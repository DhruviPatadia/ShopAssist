
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


def load_policies(policy_dir):
    """Load Markdown policy documents."""
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


def _embed_texts(client, texts, task_type):
    """Generate and normalize hosted embeddings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSIONS), dtype="float32")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    if not response.embeddings or len(response.embeddings) != len(texts):
        raise ValueError("The embedding API returned an unexpected number of vectors.")

    vectors = np.asarray(
        [item.values for item in response.embeddings],
        dtype="float32"
    )

    if vectors.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The embedding API returned a zero-length vector.")

    return np.ascontiguousarray(vectors / norms, dtype="float32")


def build_retrieval_index(documents_df, api_key=None):
    """Build a FAISS index from Gemini policy embeddings."""
    import os

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY was not provided.")

    client = genai.Client(api_key=api_key)

    embeddings = _embed_texts(
        client,
        documents_df["text"].tolist(),
        task_type="RETRIEVAL_DOCUMENT"
    )

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(embeddings)

    return client, index


def search_policies(query, documents_df, client, index, top_k=3):
    """Return the highest-ranked policy documents and cosine-style scores."""
    columns = ["rank", "source", "similarity"]

    if not query or not query.strip():
        return pd.DataFrame(columns=columns)

    if index.ntotal != len(documents_df):
        raise ValueError("Index size does not match the policy document count.")

    query_embedding = _embed_texts(
        client,
        [query.strip()],
        task_type="RETRIEVAL_QUERY"
    )

    k = min(max(int(top_k), 1), len(documents_df))
    scores, indices = index.search(query_embedding, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx >= 0:
            results.append({
                "rank": rank,
                "source": documents_df.iloc[idx]["source"],
                "similarity": float(score)
            })

    return pd.DataFrame(results, columns=columns)
