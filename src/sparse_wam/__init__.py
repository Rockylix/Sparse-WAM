"""Sparse-WAM, restricted to Cosmos3 policy inference."""

from .config import PaperConfig
from .selection import SelectionPlan, select_tokens

__all__ = ["PaperConfig", "SelectionPlan", "select_tokens"]
__version__ = "0.1.0"
