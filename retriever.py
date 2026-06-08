"""
Embed chunks into ChromaDB and retrieve top-k by cosine similarity.

The SentenceTransformer embedding function is attached to the collection so
that add() and query() embed with the SAME model automatically — we never
embed text manually here.
"""

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


def _get_collection(reset: bool = False):
    if reset:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(chunks: list[dict]) -> None:
    """Drop and recreate the collection, then add every chunk. IDs are
    f"{source}_{chunk_index}" so re-running never duplicates."""
    collection = _get_collection(reset=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for c in chunks:
        meta = c["metadata"]
        ids.append(f"{meta['source']}_{meta['chunk_index']}")
        documents.append(c["text"])
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return up to k chunks ranked by cosine distance (lower = more similar)."""
    collection = _get_collection(reset=False)
    result = collection.query(query_texts=[query], n_results=k)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "text": doc,
            "source": meta.get("source"),
            "topic": meta.get("topic"),
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


if __name__ == "__main__":
    from ingest import load_and_chunk

    print("Loading chunks...")
    chunks = load_and_chunk()
    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    embed_and_store(chunks)
    print("Done.\n")

    EVAL_QUERIES = [
        "What do r/berkeley posters say about taking CS 61A with John DeNero — is he the recommended instructor for the course?",
        "Why do students say CS 70 feels artificially hard — harder than its content warrants?",
        "What do students say about the Stat 134 Spring 2025 midterm?",
    ]

    for q in EVAL_QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        for i, r in enumerate(retrieve(q), 1):
            preview = r["text"][:100].replace("\n", " ")
            print(f"\n  [{i}] distance={r['distance']:.4f}  source={r['source']}")
            print(f"      {preview}{'...' if len(r['text']) > 100 else ''}")
        print()
