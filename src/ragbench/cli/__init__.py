"""Command line interface: one command per pipeline stage.

Errors are reported as ``error: <message>`` on stderr with a non zero exit
status. Credentials are never printed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from ..config import load_settings
from ..errors import RagBenchError
from .answer import _cmd_ask, _cmd_index, _cmd_search
from .corpus import _cmd_config, _cmd_ingest, _cmd_inspect
from .evaluation import _cmd_compare, _cmd_evaluate, _cmd_experiment
from .parser import build_parser

__all__ = ["build_parser", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "ingest": _cmd_ingest,
        "inspect": _cmd_inspect,
        "config": _cmd_config,
        "index": _cmd_index,
        "search": _cmd_search,
        "ask": _cmd_ask,
        "evaluate": _cmd_evaluate,
        "experiment": _cmd_experiment,
        "compare": _cmd_compare,
    }
    try:
        settings = load_settings(root=args.root)
        handler = handlers[args.command]
        return handler(args, settings)
    except RagBenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
