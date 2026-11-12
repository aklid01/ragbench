# RAGBench

A small, modular Retrieval-Augmented Generation (RAG) evaluation workbench.

RAGBench demonstrates and measures the complete RAG lifecycle: document
ingestion, chunking, embedding, vector retrieval, grounded answer generation
with citations, and quantitative evaluation of both retrieval and
generation.

The project is built to be read. Every stage of the pipeline is implemented
explicitly, without a framework hiding the interesting parts, and every
decision is documented where it is made.

## Status

The workbench is built in phases. This checkout is at **phase 1 (foundation)**:
configuration, ingestion, chunking and the command line for those stages.

| Phase | Content | State |
|---|---|---|
| 1 | Configuration, data models, ingestion, chunking, CLI | done |
| 2 | Embedding client, ChromaDB vector store, retrieval | pending |
| 3 | Generation client, prompt construction, citations, abstention | pending |
| 4 | Evaluation dataset, retrieval and generation metrics, experiments | pending |
| 5 | Experiment comparison, notebook, full documentation | pending |

## Features

- Ingests PDF, plain text and markdown, preserving the source file, its hash
  and its page numbers.
- Normalises extracted text (layout noise, unicode spaces, soft hyphens,
  words broken across lines).
- Cuts documents into token windows of configurable size and overlap, keeping
  the page range of every chunk.
- Detects byte duplicates and content duplicates in the corpus.
- Configurable through environment variables and a `.env` file, with secrets
  kept out of the source tree.

## Architecture

```
documents ──▶ ingest ──▶ pages ──▶ chunking ──▶ chunks
                              │                     │
                              ▼                     ▼
                        extracted store        chunk store
```

| Module | Responsibility |
|---|---|
| `ragbench.config` | Builds the one `Settings` object every other component reads |
| `ragbench.models` | `SourceRef`, `Page`, `ExtractedDocument`, `Chunk` |
| `ragbench.ingest.extract` | PDF, text and markdown extraction |
| `ragbench.ingest.normalize` | Text normalisation rules |
| `ragbench.ingest.discover` | Directory walk and duplicate detection |
| `ragbench.ingest.runner` | The ingestion pipeline |
| `ragbench.chunking.chunker` | Token window chunking with page provenance |
| `ragbench.chunking.spec` | Chunking configurations |
| `ragbench.store` | The on-disk layout of all artefacts |
| `ragbench.cli` | Command line entry point |

## Technology

- Python 3.12 or newer
- PyMuPDF for PDF extraction
- Pydantic for data validation
- ChromaDB for the vector store (phase 2)
- tiktoken for token counting
- httpx for the OpenAI-compatible API clients (phases 2 and 3)
- pytest for tests
- pandas and matplotlib for experiment analysis (phase 5)

No LLM framework is used. Document processing, chunking, embedding, retrieval,
prompt construction and answer generation are implemented in this repository.

## Installation

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"      # Windows
.venv/bin/python -m pip install -e ".[dev]"          # macOS and Linux
```

Optional extras: `[index]` for ChromaDB, `[analysis]` for pandas and
matplotlib, `[notebook]` for Jupyter.

## Configuration

Copy `.env.example` to `.env` and fill in what you need. `.env` is
gitignored. Settings are read from the environment first and from `.env`
second, so an exported variable always wins.

| Variable | Meaning | Default |
|---|---|---|
| `RAGBENCH_LLM_BASE_URL` | OpenAI-compatible generation endpoint | `http://127.0.0.1:3425/v1` |
| `RAGBENCH_LLM_MODEL` | Generation model | `minimax-code-global/MiniMax-M3.1-Flash` |
| `RAGBENCH_LLM_API_KEY` | Generation key, when the endpoint needs one | unset |
| `RAGBENCH_EMBEDDING_BASE_URL` | OpenAI-compatible embedding endpoint | `https://integrate.api.nvidia.com/v1` |
| `RAGBENCH_EMBEDDING_MODEL` | Embedding model | `nvidia/nemotron-3.5-embed-8b` |
| `RAGBENCH_EMBEDDING_API_KEY` | Embedding key | unset |
| `RAGBENCH_EMBEDDING_QUERY_INPUT_TYPE` | `input_type` for questions | `query` |
| `RAGBENCH_EMBEDDING_Passage_INPUT_TYPE` | `input_type` for passages | `passage` |
| `RAGBENCH_CHUNK_SIZE_TOKENS` | Default chunk size | `800` |
| `RAGBENCH_CHUNK_OVERLAP_TOKENS` | Default chunk overlap | `80` |
| `RAGBENCH_TOP_K` | Default number of retrieved chunks | `5` |
| `RAGBENCH_TOKENIZER_ENCODING` | Tokenizer used to measure chunk sizes | `cl100k_base` |

Chunk sizes are counted in tokens with `tiktoken`. `cl100k_base` is the
encoding OpenAI compatible models are measured with; for another provider it
is an approximation, which is why the encoding name is written into every
chunk file.

## Usage

```bash
ragbench ingest --source data/documents            # extract and chunk
ragbench ingest --source data/documents --preset small
ragbench ingest --source data/documents --chunk-size 500 --overlap 50
ragbench inspect documents                          # what was ingested
ragbench inspect chunks --doc-id <document id>     # what was cut
ragbench config                                    # effective settings, secrets hidden
```

### Chunking configurations

| Preset | Size | Overlap |
|---|---|---|
| `small` | 500 tokens | 50 tokens |
| `medium` | 800 tokens | 80 tokens |
| `large` | 1200 tokens | 120 tokens |

## Storage layout

```
data/documents/        original files (never committed)
data/extracted/        extracted text per document plus the ingest manifest
data/chunks/           chunk files, one directory per configuration
data/eval/             the evaluation dataset (committed)
artifacts/index/       the vector index
artifacts/experiments/ experiment runs
artifacts/reports/     generated reports
```

## Testing

```bash
.venv\Scripts\python -m pytest
```

The tests use mocked API responses and never call a paid endpoint.

## Limitations

- Phase 1 covers ingestion and chunking only. Retrieval, generation and
  evaluation arrive in later phases.
- Chunk sizes are measured with `cl100k_base`, which is an approximation for
  models with a different tokenizer.
- PDF extraction inherits the layout of the source file. Multi column pages
  and scanned pages produce poor text; scanned pages produce no text at all.
