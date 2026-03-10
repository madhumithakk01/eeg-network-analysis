"""
Unified model comparison: aggregate metrics from RF, Temporal CNN, Connectivity DL, and Connectivity DL (stride=4).
"""

from .comparison import run_model_comparison

__all__ = ["run_model_comparison"]
