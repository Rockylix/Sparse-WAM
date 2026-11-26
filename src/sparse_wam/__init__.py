"""Sparse-WAM inference for Cosmos3 and FastWAM-Joint policies."""

from .config import SparseWAMConfig
from .selection import SelectionPlan, select_tokens

__all__ = ["SparseWAMConfig", "SelectionPlan", "select_tokens"]
__version__ = "0.1.0"
