"""Central configuration for The Unofficial Guide RAG pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Embeddings ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Documents ---
DOCS_DIR = "documents"

# --- Vector store ---
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "unofficial_guide"

# --- Retrieval ---
TOP_K = 4
DISTANCE_THRESHOLD = 0.6

# --- Chunking ---
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
