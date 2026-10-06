"""Sparse-WAM, restricted to Cosmos3 policy inference."""

from .config import SparseWAMConfig
from .selection import SelectionPlan, select_tokens

__all__ = ["SparseWAMConfig", "SelectionPlan", "select_tokens"]
__version__ = "0.1.0"
