# RAGBench

A modular RAG evaluation workbench: document ingestion, chunking, embedding,
vector retrieval, grounded answer generation with citations, and quantitative
evaluation of retrieval and generation. No LLM framework is used; every stage
of the pipeline is implemented in this repository.

## Features

- Ingests PDF, plain text and markdown, preserving the source file, its hash
  and its page numbers.
- Normalises extracted text (layout noise, unicode spaces, soft hyphens, words
  broken across lines).
- Cuts documents into token windows of configurable size and overlap, keeping
  the page range of every chunk.
- Embeds chunks and queries through any OpenAI-compatible endpoint, with
  batching, retries and rate limiting.
- Stores vectors in a local ChromaDB collection per chunking configuration,
  with a guard against silently mixing incompatible embedding setups.
- Answers questions grounded in the retrieved passages, with bracketed
  citations validated against what was actually retrieved and a deterministic
  abstention marker when the evidence is insufficient.
- Evaluates with retrieval metrics (Recall@K, MRR@K), generation metrics
  (citation precision and recall, faithfulness, abstention, token F1) and
  operational metrics (latency, tokens, errors).
- Compares chunk sizes and top-K values in a reproducible experiment grid,
  persisting every run for audit.

## Architecture

```
documents -> ingest -> pages -> chunking -> chunks -> index -> retrieval
                                                              |
                                    evaluation <- generation <-+
```

| Package               | Responsibility                                                                   |
| --------------------- | -------------------------------------------------------------------------------- |
| `ragbench.config`     | Builds the one `Settings` object every component reads                           |
| `ragbench.models`     | `SourceRef`, `Page`, `ExtractedDocument`, `Chunk`                                |
| `ragbench.ingest`     | PDF, text and markdown extraction, normalisation, discovery, pipeline            |
| `ragbench.chunking`   | Token window chunking with page provenance and presets (500 / 800 / 1200 tokens) |
| `ragbench.embeddings` | OpenAI-compatible embedding client                                               |
| `ragbench.retrieval`  | ChromaDB index with a configuration guard, embed-once indexing, top-K search     |
| `ragbench.generation` | Prompt construction, chat client, citation validation, grounded answer pipeline  |
| `ragbench.evaluation` | Dataset loader, metrics, experiment grid, run persistence                        |
| `ragbench.cli`        | Command line entry point                                                         |

## Installation

Requires Python 3.12 or newer.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[ingest,index,analysis,notebook,dev]"   # Windows
.venv/bin/python -m pip install -e ".[ingest,index,analysis,notebook,dev]"       # macOS and Linux
```

Extras: `ingest` (PyMuPDF, tiktoken), `index` (ChromaDB), `analysis` (pandas,
matplotlib), `notebook` (JupyterLab), `dev` (pytest, pytest-cov, mypy, ruff).
`requirements-lock.txt` pins exact versions for reproducible installs.

## Configuration

Settings are read from environment variables first, then from a `.env` file at
the project root, then from defaults. Copy `.env.example` to `.env` and fill
in what you need; it documents every variable and its default.

## Usage

```bash
ragbench ingest --source data/documents            # extract and chunk
ragbench ingest --source data/documents --preset small
ragbench index --spec medium-800                   # embed chunks into the index
ragbench inspect documents                         # what was ingested
ragbench inspect chunks --doc-id <document id>     # what was cut
ragbench search "your question"                    # retrieval only
ragbench ask "your question"                       # grounded answer with citations
ragbench config                                    # effective settings, secrets hidden
```

### Chunking presets

| Preset   | Size        | Overlap    |
| -------- | ----------- | ---------- |
| `small`  | 500 tokens  | 50 tokens  |
| `medium` | 800 tokens  | 80 tokens  |
| `large`  | 1200 tokens | 120 tokens |

### Evaluation and experiments

```bash
ragbench evaluate                                  # run data/eval/dataset.json (30 cases)
ragbench evaluate --dataset my-dataset.json        # your own dataset
ragbench evaluate --spec small-500 --top-k 3       # one configuration
ragbench experiment                                # the full chunk-size x top-K grid
ragbench experiment --specs medium --top-ks 3,5    # a reduced grid
ragbench compare                                   # table of every saved run
```

The pipeline works with any documents you ingest, but evaluation needs a
dataset matched to your corpus: each case names the document and passage that
must back the expected answer (see `ragbench/evaluation/dataset.py`). The
bundled `data/eval/dataset.json` is a sample written for one example corpus;
pass your own file with `--dataset`.

Evaluation runs are persisted as JSON with every answer, citation and
retrieved chunk, so each metric can be audited by hand.
`notebooks/evaluation_analysis.ipynb` loads the saved runs and produces the
comparison charts.

## Testing

```bash
.venv/Scripts/python -m pytest    # Windows
.venv/bin/python -m pytest        # macOS and Linux
```

Tests run against fake OpenAI-compatible servers on localhost and never call a
paid endpoint. E2E tests drive the full `ingest -> index -> ask` path and the
`evaluate -> experiment -> compare` path. Lint and type check with
`ruff check .` and `mypy`.

## Limitations

- Chunk sizes are measured with `cl100k_base`, an approximation for models
  with a different tokenizer.
- PDF extraction follows the source layout: multi-column pages extract poorly
  and scanned pages produce no text.
- The bundled evaluation dataset is 30 hand-written cases over one corpus of
  six ESG reports: metrics on it measure that corpus, not RAG in general.
- Token F1 and exact match grade wording against reference answers, not truth;
  the saved answers are the ground truth for judging quality.
- Retrieval is a single dense vector search per question: no reranker, no
  hybrid keyword retrieval, no query rewriting. Quality depends entirely on
  the configured embedding model.
