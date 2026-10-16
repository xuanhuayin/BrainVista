"""BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction."""
from .models.predictor import BrainVista, build_attention_mask
from .models.tokenizer import Dec, EncAE

__version__ = "1.0.0"
__all__ = ["BrainVista", "build_attention_mask", "EncAE", "Dec"]
