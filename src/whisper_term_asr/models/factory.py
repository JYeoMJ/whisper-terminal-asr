"""
Factory module for creating ASR models.
"""

from typing import Dict, Any, Optional, List, Tuple
import importlib
import logging

from .base import ASRModel

# Initialize logging
logger = logging.getLogger(__name__)

# Define model types
MODEL_TYPES = {
    "openai": {
        "name": "OpenAI Whisper",
        "description": "Original OpenAI Whisper model",
        "sizes": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        "module": "whisper_term_asr.models.whisper_openai",
        "class": "OpenAIWhisperModel"
    },
    "faster": {
        "name": "Faster Whisper",
        "description": "CTranslate2-based Whisper implementation optimized for CPU",
        "sizes": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        "module": "whisper_term_asr.models.whisper_faster",
        "class": "FasterWhisperModel"
    },
    "insanely-fast": {
        "name": "Insanely Fast Whisper",
        "description": "Whisper implementation with Flash Attention 2 and BetterTransformer",
        "sizes": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        "module": "whisper_term_asr.models.whisper_insanely_fast",
        "class": "InsanelyFastWhisperModel"
    },
    "mlx": {
        "name": "MLX Whisper",
        "description": "Apple MLX-based Whisper implementation optimized for Apple Silicon",
        "sizes": ["tiny", "base", "small", "medium", "large"],
        "module": "whisper_term_asr.models.whisper_mlx",
        "class": "MLXWhisperModel"
    },
    "lightning-mlx": {
        "name": "Lightning Whisper MLX",
        "description": "Ultra-fast Whisper implementation optimized for Apple Silicon",
        "sizes": ["tiny", "small", "distil-small.en", "base", "medium", "distil-medium.en", 
                 "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"],
        "module": "whisper_term_asr.models.whisper_lightning_mlx",
        "class": "LightningWhisperMLXModel"
    }
}


def create_model(
    model_type: str = "faster",
    model_size: str = "base",
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: str = "float16",
    cache_dir: Optional[str] = None,
    batch_size: int = 12,
    quantization: Optional[str] = None,
    **kwargs
) -> ASRModel:
    """
    Create an ASR model from the specified type and parameters.
    
    Args:
        model_type: Type of ASR model (openai, faster, insanely-fast, mlx, lightning-mlx)
        model_size: Size of the model (tiny, base, small, medium, large, distil-*)
        language: Language code (e.g., 'en')
        device: Device to run on (cpu, cuda, mps)
        compute_type: Computation type/precision (float32, float16, int8)
        cache_dir: Directory to cache models
        batch_size: Batch size (primarily for Lightning MLX)
        quantization: Quantization level (for Lightning MLX)
        **kwargs: Additional arguments for specific model implementations
        
    Returns:
        An instance of the requested ASR model
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(MODEL_TYPES.keys())}"
        )
    
    # Get model info
    model_info = MODEL_TYPES[model_type]
    
    # Import the module and class dynamically
    try:
        module = importlib.import_module(model_info["module"])
        model_class = getattr(module, model_info["class"])
    except ImportError as e:
        raise ImportError(
            f"Failed to import {model_info['name']}. "
            f"You might need to install the required dependencies: {str(e)}"
        )
    
    # Create the model instance
    model = model_class(
        model_size=model_size,
        language=language,
        device=device,
        compute_type=compute_type,
        cache_dir=cache_dir,
        batch_size=batch_size,
        quantization=quantization,
        **kwargs
    )
    
    return model


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List available ASR models with their details.
    
    Returns:
        Dictionary of model types with their details
    """
    return MODEL_TYPES
