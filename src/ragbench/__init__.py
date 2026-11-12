"""RAGBench: a small, modular retrieval augmented generation workbench.

The package is organised along the pipeline:

``ragbench.config``      one settings object for the whole run
``ragbench.models``      the data that travels through the pipeline
``ragbench.ingest``      documents in, pages of text out
``ragbench.chunking``    pages in, retrievable chunks out
``ragbench.store``       the on-disk layout of everything above
``ragbench.cli``         the command line entry point
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
