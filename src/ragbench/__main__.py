"""Module entry point, so ``python -m ragbench`` runs the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
