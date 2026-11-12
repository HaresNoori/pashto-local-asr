"""SHA-256 of an input file, streamed so large audio does not load into memory."""
import hashlib
from pathlib import Path


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
