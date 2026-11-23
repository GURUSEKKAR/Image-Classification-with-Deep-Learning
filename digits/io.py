"""Publish an output only after its writer finishes successfully."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        pending = Path(temp.name)
    try:
        yield pending
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def report(result, path):
    output = json.dumps(result, indent=2, allow_nan=False)
    if path is None or str(path) == "-":
        print(output)
        return
    with atomic_output(path) as pending:
        pending.write_text(output + "\n", encoding="utf-8")
    print(f"Wrote {path}")
