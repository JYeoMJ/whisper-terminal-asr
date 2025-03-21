"""
Lightning Whisper MLX model implementation.
Uses the Lightning Whisper MLX implementation optimized for Apple Silicon.
"""

import os
import time
import numpy as np
import tempfile
import soundfile as sf
import re
from typing import Dict, Any, Optional, Union, List

try:
    from lightning_whisper_mlx import LightningWhisperMLX
    HAS_LIGHTNING_MLX = True
except ImportError:
    HAS_LIGHTNING_MLX = False

from .base import ASRModel


class LightningWhisperMLXModel(ASRModel):
    """
    Implementation of Lightning Whisper using MLX.
    Provides extremely fast inference on Apple Silicon (M1/M2/M3).
    """
    
    AVAILABLE_SIZES = ["tiny", "small", "distil-small.en", "base", "medium", 
                       "distil-medium.en", "large", "large-v2", "distil-large-v2", 
                       "large-v3", "distil-large-v3"]
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,  # Not used for MLX but kept for API consistency
        compute_type: str = "float16",  # Default to float16 for MLX
        cache_dir: Optional[str] = None,
        batch_size: int = 12,
        quantization: Optional[str] = None,
    ):
        """
        Initialize the Lightning Whisper MLX model.
        
        Args:
            model_size: Model size (tiny, small, distil-small.en, base, medium, 
                       distil-medium.en, large, large-v2, distil-large-v2, 
                       large-v3, distil-large-v3)
            language: Language code (e.g., 'en', 'fr', None for auto) - may be ignored 
                     if using language-specific distilled models
            device: Not used for MLX (kept for API consistency)
            compute_type: Computation type/precision (float16 recommended for MLX)
            cache_dir: Directory to cache models
            batch_size: Batch size for processing audio chunks (higher is better for 
                       throughput but might cause memory issues)
            quantization: Quantization level (None, "4bit", "8bit")
        """
        if not HAS_LIGHTNING_MLX:
            raise ImportError(
                "Lightning Whisper MLX is required. "
                "Install with: pip install lightning-whisper-mlx"
            )
            
        # Use the base class init but override the available sizes
        super().__init__(
            model_size=model_size,
            language=language,
            device="mps",  # Lightning MLX always uses MPS/Metal
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        if model_size not in self.AVAILABLE_SIZES:
            raise ValueError(f"Model size must be one of {self.AVAILABLE_SIZES}")
            
        self.batch_size = batch_size
        self.quantization = quantization
        
        # Validate quantization parameter
        if quantization not in [None, "4bit", "8bit"]:
            raise ValueError("Quantization must be None, '4bit', or '8bit'")
    
    def load_model(self):
        """Load the Lightning Whisper MLX model."""
        if self.model is not None:
            return
        
        # Set cache directory if provided
        if self.cache_dir:
            os.environ["HF_HOME"] = self.cache_dir
        
        # Lightning Whisper MLX handles Apple Silicon optimizations automatically
        self.model = LightningWhisperMLX(
            model=self.model_size,
            batch_size=self.batch_size,
            quant=self.quantization
        )
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Lightning Whisper MLX.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            prompt: Optional prompt to guide transcription (may not be supported)
            
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
            # This uses the numpy property that NaN != NaN
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
        if prompt:
            # Check if initial prompt is supported
            try:
                options["initial_prompt"] = prompt
            except:
                pass  # If not supported, just ignore
        
        # Perform transcription
        start_time = time.time()
        result = self.model.transcribe(audio_path=audio_path)
        inference_time = time.time() - start_time
        
        # Clean up temporary file if created
        if isinstance(audio, np.ndarray) and 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Get the transcription text and filter out problematic outputs
        text = result.get("text", "")
        
        # Only filter out strings that are MOSTLY exclamation marks (>80% of the string)
        if text.strip():
            exclamation_count = text.count('!')
            if exclamation_count > 0 and exclamation_count / len(text.strip()) > 0.8:
                # If it's just a string of mostly exclamation marks, replace with empty string
                text = ""
        
        # Format result similar to the other implementations
        formatted_result = {
            "text": text,
            "segments": [],  # Lightning MLX might not provide segments by default
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
        
    def cleanup(self):
        """
        Release resources for Lightning Whisper MLX model.
        
        This implementation focuses on properly cleaning up MLX-specific resources
        while preserving downloaded model weights.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Store a reference to the model before setting it to None
            model = self.model
            self.model = None
            
            # If the model has a specific cleanup method, call it
            if hasattr(model, 'unload') and callable(getattr(model, 'unload')):
                try:
                    model.unload()
                except Exception:
                    pass
            
            # For MLX specifically, try to release MPS memory
            try:
                import mlx.core as mx
                mx.clear_memory_pool()  # Release GPU memory
            except Exception:
                pass
                
        # Call parent cleanup to handle garbage collection
        super().cleanup()
