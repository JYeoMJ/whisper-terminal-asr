"""
Insanely Fast Whisper model implementation.
Uses optimized pipeline from Hugging Face Transformers with Flash Attention 2.
"""

import os
import time
import numpy as np
import tempfile
import soundfile as sf
from typing import Dict, Any, Optional, Union, List

try:
    import torch
    from transformers import pipeline
    from optimum.bettertransformer import BetterTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .base import ASRModel


class InsanelyFastWhisperModel(ASRModel):
    """
    Implementation of Insanely Fast Whisper.
    Combines multiple optimizations from Hugging Face ecosystem.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float16",
        cache_dir: Optional[str] = None,
        use_bettertransformer: bool = True,
        use_flash_attention: bool = True,
        batch_size: int = 16,
        chunk_length_s: int = 30,
    ):
        """
        Initialize the Insanely Fast Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'fr', None for auto)
            device: Device to run on ('cpu', 'cuda', 'mps', None for auto)
            compute_type: Computation type/precision (float32, float16, int8)
            cache_dir: Directory to cache models
            use_bettertransformer: Whether to use BetterTransformer for optimizations
            use_flash_attention: Whether to use Flash Attention 2 (if available)
            batch_size: Batch size for processing audio chunks
            chunk_length_s: Chunk length in seconds for processing long audio
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers, optimum, and accelerate are required for InsanelyFastWhisperModel. "
                "Install with: pip install transformers optimum accelerate"
            )
            
        super().__init__(
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        # Map model size to Hugging Face model names
        self.model_map = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
        }
        
        self.use_bettertransformer = use_bettertransformer
        self.use_flash_attention = use_flash_attention
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
    
    def load_model(self):
        """Load the Insanely Fast Whisper model."""
        if self.model is not None:
            return
            
        model_name = self.model_map.get(self.model_size)
        if not model_name:
            raise ValueError(f"Invalid model size: {self.model_size}")
        
        # Handle device mapping
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Check for MPS (Apple Silicon)
            if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        
        # Set cache directory if provided
        cache_dir = self.cache_dir if self.cache_dir else None
        
        # Map compute type to torch dtype
        torch_dtype = torch.float32
        if self.compute_type == "float16":
            torch_dtype = torch.float16
        elif self.compute_type == "int8":
            torch_dtype = torch.int8
        
        # Configure pipeline options
        pipe_kwargs = {
            "model": model_name,
            "device": device,
            "torch_dtype": torch_dtype,
            "model_kwargs": {"cache_dir": cache_dir},
        }
        
        # Enable Flash Attention 2 if available and requested
        if self.use_flash_attention and device != "cpu":
            try:
                from optimum.bettertransformer import BetterTransformer
                pipe_kwargs["model_kwargs"]["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except (ImportError, AttributeError):
                print("Flash Attention 2 not available, falling back to standard attention")
        
        # Create the pipeline
        self.model = pipeline(
            task="automatic-speech-recognition",
            **pipe_kwargs
        )
        
        # Apply BetterTransformer if requested
        if self.use_bettertransformer and device != "cpu":
            try:
                self.model.model = BetterTransformer.transform(self.model.model)
                print("Using BetterTransformer optimizations")
            except Exception as e:
                print(f"Failed to apply BetterTransformer: {e}")
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Insanely Fast Whisper.
        
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
            # For file paths
            audio_path = audio
        else:
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
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language
        
        if prompt:
            generate_kwargs["prompt"] = prompt
        
        # Perform transcription
        start_time = time.time()
        result = self.model(
            audio_path,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=True,
            generate_kwargs=generate_kwargs
        )
        inference_time = time.time() - start_time
        
        # Clean up temporary file if created
        if isinstance(audio, np.ndarray) and 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Format result similar to OpenAI Whisper
        formatted_result = {
            "text": result["text"],
            "inference_time": inference_time,
            "language": self.language,  # May not be accurate if auto-detected
        }
        
        # Add chunks/segments if available
        if "chunks" in result:
            formatted_result["segments"] = []
            for i, chunk in enumerate(result["chunks"]):
                formatted_result["segments"].append({
                    "id": i,
                    "start": chunk["timestamp"][0] if isinstance(chunk["timestamp"], list) else None,
                    "end": chunk["timestamp"][1] if isinstance(chunk["timestamp"], list) else None,
                    "text": chunk["text"]
                })
        
        return formatted_result
