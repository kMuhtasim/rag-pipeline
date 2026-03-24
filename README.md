# rag-pipeline

A RAG pipeline built from scratch, improving progressively.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline starting from first principles — manual chunking, cosine similarity, and OpenAI embeddings — I plan to integrate more robust vector storage backends like ChromaDB and FAISS progressively.

## How It Works

1. **Chunking** — the input text is split into overlapping chunks
2. **Embedding** — chunks are embedded using OpenAI's `text-embedding-3-small`
3. **Retrieval** — a query is embedded and compared against stored chunks using cosine similarity
4. **Generation** — the top-k retrieved chunks are passed to a language model to generate an answer

## Project Structure

```
rag-pipeline/
├── main.py                 # Entry point
├── chunk_embeddings.pkl    # Cached embeddings (auto-generated)
├── book1.txt               # Input document
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API key

### Installation

```bash
git clone https://github.com/kMuhtasim/rag-pipeline.git
cd rag-pipeline
pip install openai
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

On first run, embeddings are computed and cached to `chunk_embeddings.pkl`. Subsequent runs load from cache.

## Roadmap

- [x] Manual cosine similarity retrieval
- [ ] ChromaDB integration
- [ ] FAISS integration
- [ ] Support for multiple documents
- [ ] Evaluation metrics (e.g. hit rate, MRR)

## Dependencies

- [openai](https://github.com/openai/openai-python)

## License

MIT
