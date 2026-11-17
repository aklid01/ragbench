# RAGBench

A small, modular Retrieval-Augmented Generation (RAG) evaluation workbench.

RAGBench demonstrates and measures the complete RAG lifecycle: document
ingestion, chunking, embedding, vector retrieval, grounded answer generation
with citations, and quantitative evaluation of both retrieval and
generation.

The project is built to be read. Every stage of the pipeline is implemented
explicitly, without a framework hiding the interesting parts, and every
decision is documented where it is made.

## Features

- Ingests PDF, plain text and markdown, preserving the source file, its hash
  and its page numbers.
- Normalises extracted text (layout noise, unicode spaces, soft hyphens,
  words broken across lines).
- Cuts documents into token windows of configurable size and overlap, keeping
  the page range of every chunk.
- Embeds chunks and queries through any OpenAI-compatible endpoint, with
  batching, retries, rate limiting and content-screening handling.
- Stores vectors in a local ChromaDB collection per chunking configuration,
  with a guard against silently mixing incompatible embedding setups.
- Answers questions grounded in the retrieved passages, with bracketed
  citations validated against what was actually retrieved and a deterministic
  abstention marker when the evidence is insufficient.
- Evaluates a manually verified dataset with retrieval metrics (Recall@K,
  MRR@K), generation metrics (citation precision and recall, faithfulness,
  abstention, token F1) and operational metrics (latency, tokens, errors).
- Compares chunk sizes and top-K values in a reproducible experiment grid,
  persisting every run for audit and later analysis.

## Status

Complete: all five phases of the implementation plan are built and the
experiment grid has been run on the real corpus.

| Phase | Content | State |
|---|---|---|
| 1 | Configuration, data models, ingestion, chunking, CLI | done |
| 2 | Embedding client, ChromaDB vector store, retrieval | done |
| 3 | Generation client, prompt construction, citations, abstention | done |
| 4 | Evaluation dataset, retrieval and generation metrics, experiments | done |
| 5 | Experiment comparison, notebook, full documentation | done |

## Architecture

```
documents ──▶ ingest ──▶ pages ──▶ chunking ──▶ chunks ──▶ index ──▶ retrieval
                                                                     │
                                        evaluation ◀── generation ◀──┘
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
| `ragbench.chunking.spec` | Chunking presets (500 / 800 / 1200 tokens) |
| `ragbench.embeddings.client` | OpenAI-compatible embedding client |
| `ragbench.retrieval.store` | The ChromaDB vector index with a configuration guard |
| `ragbench.retrieval.indexer` | Embed-once incremental indexing |
| `ragbench.retrieval.search` | Top-K similarity retrieval |
| `ragbench.generation.prompts` | The grounded answer prompt and abstention marker |
| `ragbench.generation.client` | OpenAI-compatible chat client with retries |
| `ragbench.generation.citations` | Citation extraction and validation |
| `ragbench.generation.pipeline` | Retrieval to structured, cited answer |
| `ragbench.evaluation.dataset` | The eval dataset schema and loader |
| `ragbench.evaluation.runner` | Runs the dataset, scores records, persists runs |
| `ragbench.evaluation.metrics` | Recall@K, MRR@K, citation, faithfulness, abstention |
| `ragbench.evaluation.experiments` | The chunk-size x top-K grid |
| `ragbench.store` | The on-disk layout of all artefacts |
| `ragbench.cli` | Command line entry point |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design decisions
and [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) for a guided tour.

## Technology

- Python 3.12 or newer
- Pydantic for data validation
- httpx for the OpenAI-compatible API clients
- PyMuPDF (ingest extra) for PDF extraction
- tiktoken (ingest extra) for token counting
- ChromaDB (index extra) for the vector store
- pytest for tests
- pandas and matplotlib (analysis extra) for experiment analysis
- Jupyter (notebook extra) for the evaluation notebook

No LLM framework is used. Document processing, chunking, embedding, retrieval,
prompt construction and answer generation are implemented in this repository.

## Installation

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[ingest,index,analysis,notebook,dev]"   # Windows
.venv/bin/python -m pip install -e ".[ingest,index,analysis,notebook,dev]"       # macOS and Linux
```

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
| `RAGBENCH_EMBEDDING_PASSAGE_INPUT_TYPE` | `input_type` for passages | `passage` |
| `RAGBENCH_EMBEDDING_DIMENSIONS` | Expected vector size, validated on every response | unset |
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
ragbench index --spec medium-800                   # embed chunks into the index
ragbench inspect documents                         # what was ingested
ragbench inspect chunks --doc-id <document id>     # what was cut
ragbench search "what is the net zero target?"     # retrieval only
ragbench ask "what is the net zero target?"        # grounded answer with citations
ragbench config                                    # effective settings, secrets hidden
```

### Chunking configurations

| Preset | Size | Overlap |
|---|---|---|
| `small` | 500 tokens | 50 tokens |
| `medium` | 800 tokens | 80 tokens |
| `large` | 1200 tokens | 120 tokens |

### Evaluation and experiments

```bash
ragbench evaluate                                  # run data/eval/dataset.json
ragbench evaluate --spec small-500 --top-k 3       # one configuration
ragbench experiment                                # the full chunk-size x top-K grid
ragbench experiment --specs medium --top-ks 3,5    # a reduced grid
ragbench compare                                   # table of every saved run
```

Evaluation runs are stored under `artifacts/experiments/runs/` as JSON with
every answer, citation and retrieved chunk, so each metric can be audited by
hand. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) records the actual results.
`notebooks/evaluation_analysis.ipynb` loads every saved run and produces the
comparison charts in `artifacts/reports/`.

## Storage layout

```
data/documents/        original files (never committed)
data/extracted/        extracted text per document plus the ingest manifest
data/chunks/           chunk files, one directory per configuration
data/eval/             the evaluation dataset (committed)
artifacts/index/       the vector index, one directory per configuration
artifacts/experiments/ experiment runs
artifacts/reports/     generated charts and reports
```

## Testing

```bash
.venv\Scripts\python -m pytest
```

The tests use fake OpenAI-compatible servers on localhost and never call a
paid endpoint. One E2E test drives the full `ingest -> index -> ask` path and
another drives `evaluate -> experiment -> compare` with the same fakes.

## Limitations

- Chunk sizes are measured with `cl100k_base`, which is an approximation for
  models with a different tokenizer.
- PDF extraction inherits the layout of the source file. Multi column pages
  and scanned pages produce poor text; scanned pages produce no text at all.
- The evaluation dataset is small (15 cases written by hand, expanded to 30+
  during verification) and covers one corpus of six ESG reports; metrics on
  it measure this corpus, not RAG in general.
- Token F1 and exact match are lexical comparisons against reference answers:
  they grade wording, not truth. The saved answers, not the metrics, are the
  ground truth for judging quality.
- Retrieval quality depends entirely on the configured embedding model. The
  experiments use `nvidia/nemotron-3.5-embed-8b`; a different model changes
  every retrieval number.
- No reranker, no hybrid keyword retrieval and no query rewriting: retrieval
  is a single dense vector search per question.
