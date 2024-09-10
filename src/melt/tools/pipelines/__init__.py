"""
This module provides the EvalPipeline and MetricPipeline classes 
for evaluation and metric processing in machine learning workflows.
"""
from .pipelines import EvalPipeline
from .metric_pipelines import MetricPipeline


__all__ = ["EvalPipeline", "MetricPipeline"]
