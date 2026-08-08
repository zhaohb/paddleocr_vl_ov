"""
PaddleOCR-VL Pipeline 模块
"""

SUPPORTED_MODELS = ["PP-LCNet_x1_0_doc_ori", "UVDoc"]

from .ov_paddleocr_vl_pipeline import PaddleOCRVL, PaddleOCRVLResult, PaddleOCRVLBlock

__all__ = ['PaddleOCRVL', 'PaddleOCRVLResult', 'PaddleOCRVLBlock']
