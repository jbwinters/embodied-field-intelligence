#!/usr/bin/env python3
"""Convenience shim so `python cli.py ...` works from the repo root.

The actual CLI lives in efi.cli (also available as the `efi` console
script after `pip install -e .`).
"""

from efi.cli import main

if __name__ == "__main__":
    main()
