"""The retrieval and answering commands: index, search and ask."""

from __future__ import annotations

import argparse
import json

from ..config import Settings
from ..embeddings import EmbeddingClient
from ..errors import RetrievalError
from ..generation import AnsweredQuestion, LLMClient, answer_question
from ..retrieval.indexer import index_chunks, latest_indexed_spec
from ..retrieval.search import search_chunks
from .common import _prompt_questions, _resolve_chunk_spec


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    spec_name = _resolve_chunk_spec(args.spec, settings)
    client = EmbeddingClient(settings.embedding)
    try:
        summary = index_chunks(paths, settings.embedding, client, spec_name, rebuild=args.rebuild)
    finally:
        client.close()

    if args.json:
        print(
            json.dumps(
                {
                    "spec": summary.spec_name,
                    "embedded": summary.embedded,
                    "skipped": summary.skipped,
                    "rejected": summary.rejected,
                    "pruned": summary.pruned,
                    "vector_count": summary.vector_count,
                    "dimensions": summary.dimensions,
                    "failures": summary.failures,
                    "warnings": summary.warnings,
                },
                indent=2,
            )
        )
    else:
        print(f"indexed: {summary.describe()}")
        for warning in summary.warnings:
            print(f"  warning: {warning}")
        for failure in summary.failures:
            print(f"  failure: {failure}")
    return 0


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    spec_name = args.spec or latest_indexed_spec(paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    top_k = args.top_k or settings.retrieval.top_k
    client = EmbeddingClient(settings.embedding)
    try:
        results = search_chunks(paths, settings.embedding, client, spec_name, args.query, top_k)
    finally:
        client.close()

    if args.json:
        payload = {
            "query": args.query,
            "spec": spec_name,
            "top_k": top_k,
            "model": settings.embedding.model,
            "results": [result.model_dump() for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"query: {args.query}  (configuration {spec_name}, top {top_k})")
        for result in results:
            print(
                f"{result.rank}. distance={result.distance:.4f}  {result.citation_label}  "
                f"{result.chunk_id}"
            )
            excerpt = result.text[:160] + ("..." if len(result.text) > 160 else "")
            print(f"    {excerpt}")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    spec_name = args.spec or latest_indexed_spec(paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    top_k = args.top_k or settings.retrieval.top_k

    questions = [args.question] if args.question else _prompt_questions()
    embedding = EmbeddingClient(settings.embedding)
    llm = LLMClient(settings.llm)
    try:
        for question in questions:
            result = answer_question(
                paths,
                settings.embedding,
                embedding,
                settings.llm,
                llm,
                spec_name,
                question,
                top_k,
            )
            if args.json:
                print(json.dumps(result.model_dump(mode="json"), indent=2))
            else:
                _print_answer(result, spec_name, top_k)
    finally:
        embedding.close()
        llm.close()
    return 0


def _print_answer(result: AnsweredQuestion, spec_name: str, top_k: int) -> None:
    print(f"question: {result.question}  (configuration {spec_name}, top {top_k})")
    if result.abstained:
        print("answer: <abstained> the retrieved passages do not contain enough information")
    else:
        print(f"answer: {result.answer}")
    for citation in result.citations:
        print(f"  [{citation.reference}] {citation.citation_label}  {citation.chunk_id}")
        excerpt = citation.excerpt[:160] + ("..." if len(citation.excerpt) > 160 else "")
        print(f"      {excerpt}")
    if result.invalid_references:
        print(f"  invalid references with no matching passage: {result.invalid_references}")
    if result.total_tokens is not None:
        tokens = (
            f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
            f"total={result.total_tokens}"
        )
    else:
        tokens = "usage not reported"
    print(
        f"model: {result.model or 'unknown'}  tokens: {tokens}  "
        f"retrieval: {result.retrieval_seconds:.2f}s  generation: {result.generation_seconds:.2f}s"
    )
