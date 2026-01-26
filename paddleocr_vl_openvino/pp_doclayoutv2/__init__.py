"""
PP-DocLayoutV2 Layout Detection Module

This module provides OpenVINO-based inference for PP-DocLayoutV2 document layout detection.
"""

from .ov_pp_layoutv2_infer import (
    paddle_ov_doclayout,
    preprocess_image_doclayout,
    postprocess_detections_detr,
)

# Export result types from `result.py` (do not depend on infer module import order)
from .result import LayoutAnalysisResult

# Backward-compat alias
LayoutDetectionResult = LayoutAnalysisResult

__all__ = [
    'paddle_ov_doclayout',
    'preprocess_image_doclayout',
    'postprocess_detections_detr',
    'LayoutDetectionResult',
    'LayoutAnalysisResult',
]

