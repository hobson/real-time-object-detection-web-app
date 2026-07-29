"""Short, stable identity for a model checkpoint file - used to key
LabelSource rows (see inference-server/orm.py's LabelSource docstring) so
re-scoring the same checkpoint reuses the same row instead of creating a
new identity every run.
"""
import hashlib
from pathlib import Path

DEFAULT_HASH_PREFIX_LEN = 12


def weights_hash(path: str | Path, prefix_len: int = DEFAULT_HASH_PREFIX_LEN) -> str:
    """sha256 of the checkpoint file's bytes, truncated to prefix_len hex
    chars - short enough to store/display alongside model_name, long enough
    that two different checkpoints collide only by astronomical chance."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:prefix_len]
