# rag-pipeline

A RAG pipeline built from scratch, improving progressively.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline starting from first principles — manual chunking, OpenAI embeddings, and a vector store for retrieval. Storage and nearest-neighbour search are handled by **ChromaDB**, replacing the earlier pickle-based cache and manual cosine similarity approach.

## How It Works

1. **Chunking** — the input text is split into overlapping chunks
2. **Embedding** — chunks are embedded using OpenAI's `text-embedding-3-small`
3. **Storage** — embeddings and documents are persisted in a local ChromaDB collection (`chroma_db/`)
4. **Retrieval** — a query is embedded and the top-k most similar chunks are fetched via ChromaDB's ANN (Approximate Nearest Neighbour) search using cosine distance
5. **Generation** — the retrieved chunks are passed to a language model to generate an answer

## Project Structure

```
rag-pipeline/
├── main.py                 # Entry point
├── book1.txt               # Input document
├── chroma_db/              # Persistent ChromaDB storage (auto-generated)
└── README.md
```

> `chunk_embeddings.pkl` is no longer used and has been removed. ChromaDB handles persistence automatically.

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API key

### Installation

```bash
git clone https://github.com/kMuhtasim/rag-pipeline.git
cd rag-pipeline
pip install openai chromadb
```

### Usage

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Place your text file as `book1.txt` in the root directory, then run:

```bash
python main.py
```

On first run, chunks are embedded and stored in the local `chroma_db/` directory. Subsequent runs load directly from ChromaDB — no re-embedding needed.

## Storage & Retrieval

| | Previous | Current |
|---|---|---|
| **Storage** | `chunk_embeddings.pkl` (pickle) | `chroma_db/` (ChromaDB persistent client) |
| **Search** | Exact nearest neighbour (O(n) cosine similarity) | Approximate nearest neighbour (HNSW via ChromaDB) |
| **Distance metric** | Cosine similarity | Cosine distance |

## Roadmap

- [x] Manual cosine similarity retrieval
- [x] ChromaDB integration
- [ ] FAISS integration
- [ ] Support for multiple documents
- [ ] Evaluation metrics (e.g. hit rate, MRR)

## Dependencies

- [openai](https://github.com/openai/openai-python)
- [chromadb](https://github.com/chroma-core/chroma)

## License

MIT
