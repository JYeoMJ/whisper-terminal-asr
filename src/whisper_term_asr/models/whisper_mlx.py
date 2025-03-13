"""
MLX Whisper model implementation.
Uses MLX framework optimized for Apple Silicon (M1/M2/M3 chips).
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional, Union, List
import re
import tempfile
import soundfile as sf

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .base import ASRModel


class MLXWhisperModel(ASRModel):
    """
    Implementation of Whisper using MLX framework.
    Optimized for Apple Silicon (M1/M2/M3) with remarkable performance on Mac.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,  # Not used for MLX but kept for API consistency
        compute_type: str = "float16",  # Default to float16 for MLX
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the MLX Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'fr', None for auto)
            device: Not used for MLX (kept for API consistency)
            compute_type: Computation type/precision (float16 recommended for MLX)
            cache_dir: Directory to cache models
        """
        if not HAS_MLX:
            raise ImportError(
                "MLX and MLX-LM are required for MLXWhisperModel. "
                "Install with: pip install mlx mlx-lm"
            )
            
        super().__init__(
            model_size=model_size,
            language=language,
            device="mps",  # MLX always uses MPS/Metal
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        # Map model size to MLX-compatible model names
        self.model_map = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
        }
    
    def load_model(self):
        """Load the MLX Whisper model."""
        if self.model is not None:
            return
            
        model_name = self.model_map.get(self.model_size)
        if not model_name:
            raise ValueError(f"Invalid model size: {self.model_size}")
        
        # Set cache directory if provided
        cache_dir = self.cache_dir if self.cache_dir else None
        
        # Load the model - MLX will handle optimizations for Apple Silicon
        self.model = load(
            model_name,
            cache_dir=cache_dir,
            dtype=mx.float16 if self.compute_type == "float16" else mx.float32
        )
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using MLX Whisper.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dictionary containing transcription results
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Handle different audio input types
        if isinstance(audio, str):
            # File path - can be handled directly
            audio_path = audio
        else:
            # Fix NaN values in audio data that can cause exclamation mark outputs
            if np.issubdtype(audio.dtype, np.floating) and np.any(audio != audio):
                audio = audio.copy()  # Create a copy to avoid modifying the original
                audio[audio != audio] = 0.0  # Replace NaN values with zeros
                
            # For numpy arrays, we need to save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # If audio is float and already normalized between -1 and 1, save directly
            if np.issubdtype(audio.dtype, np.floating) and np.max(np.abs(audio)) <= 1.0:
                sf.write(temp_path, audio, 16000)
            else:
                # Convert to float32 and normalize
                audio_norm = audio.astype(np.float32) / np.iinfo(np.int16).max
                sf.write(temp_path, audio_norm, 16000)
                
            audio_path = temp_path
        
        # Set transcription options
        options = {}
        if self.language:
            options["language"] = self.language
            
        if prompt:
            options["prompt"] = prompt
            
        # Perform transcription
        start_time = time.time()
        
        # MLX Whisper requires us to use the specific transcribe function
        result = self.model.transcribe(audio_path, **options)
        inference_time = time.time() - start_time
        
        # Clean up temporary file if created
        if isinstance(audio, np.ndarray) and 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Filter out problematic exclamation marks in the main text
        text = result.get("text", "")
        # Only filter if the text is mostly exclamation marks
        if text.strip():
            exclamation_count = text.count('!')
            if exclamation_count > 0 and exclamation_count / len(text.strip()) > 0.8:
                text = ""
            
        # Format result similar to the other implementations
        formatted_result = {
            "text": text,
            "segments": [],  # MLX might not provide segments by default
            "language": self.language,  # May not be accurate
            "inference_time": inference_time
        }
        
        # Add segments if available
        if "segments" in result:
            # Also filter exclamation marks in segments
            clean_segments = []
            for segment in result["segments"]:
                if "text" in segment:
                    segment_text = segment["text"]
                    # Only filter if the segment is mostly exclamation marks
                    if segment_text.strip():
                        exclamation_count = segment_text.count('!')
                        if exclamation_count > 0 and exclamation_count / len(segment_text.strip()) > 0.8:
                            segment_text = ""
                    segment["text"] = segment_text
                clean_segments.append(segment)
            formatted_result["segments"] = clean_segments
        else:
            formatted_result["segments"] = result.get("segments", [])
        
        return formatted_result
