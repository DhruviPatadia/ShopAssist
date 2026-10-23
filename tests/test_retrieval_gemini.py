
import numpy as np
import pandas as pd
import pytest

from src.retrieval_gemini import (
    _embed_texts,
    search_policies,
    EMBEDDING_DIMENSIONS,
)


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, vectors):
        self.embeddings = [FakeEmbedding(v) for v in vectors]


class FakeModels:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.vectors)


class FakeClient:
    def __init__(self, vectors):
        self.models = FakeModels(vectors)


def test_embeddings_are_normalized():
    client = FakeClient([[3.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)])

    vectors = _embed_texts(client, ["test"], "RETRIEVAL_QUERY")

    assert vectors.shape == (1, EMBEDDING_DIMENSIONS)
    assert vectors.dtype == np.float32
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_empty_embedding_input():
    client = FakeClient([])

    vectors = _embed_texts(client, [], "RETRIEVAL_DOCUMENT")

    assert vectors.shape == (0, EMBEDDING_DIMENSIONS)
    assert not client.models.calls


def test_search_returns_matching_source_first():
    documents = pd.DataFrame({
        "source": ["returns_policy.md", "delivery_policy.md"],
        "text": ["Returns information", "Delivery information"],
    })

    query_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    other_vector = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)

    index_vectors = np.asarray(
        [query_vector, other_vector], dtype=np.float32
    )
    import faiss
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(index_vectors)

    client = FakeClient([query_vector])

    results = search_policies(
        "How do returns work?", documents, client, index, top_k=2
    )

    assert results.iloc[0]["source"] == "returns_policy.md"
    assert len(results) == 2
    assert results.iloc[0]["similarity"] == pytest.approx(1.0)


def test_empty_query_returns_no_results():
    documents = pd.DataFrame({
        "source": ["returns_policy.md"],
        "text": ["Returns information"],
    })

    import faiss
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(np.zeros((1, EMBEDDING_DIMENSIONS), dtype=np.float32))
    client = FakeClient([])

    results = search_policies("", documents, client, index)

    assert results.empty
    assert list(results.columns) == ["rank", "source", "similarity"]


def test_index_size_mismatch_raises_error():
    documents = pd.DataFrame({
        "source": ["returns_policy.md", "delivery_policy.md"],
        "text": ["Returns", "Delivery"],
    })

    import faiss
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(np.zeros((1, EMBEDDING_DIMENSIONS), dtype=np.float32))
    client = FakeClient([])

    with pytest.raises(ValueError, match="Index size"):
        search_policies("returns", documents, client, index)
