"""
OpenAI Whisper model implementation.
Uses the original Whisper model from OpenAI.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional, Union
import re

import whisper

from .base import ASRModel


class OpenAIWhisperModel(ASRModel):
    """
    Implementation of the original OpenAI Whisper model.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float32",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the OpenAI Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'fr', None for auto)
            device: Device to run on ('cpu', 'cuda', 'mps', None for auto)
            compute_type: Computation type/precision (float32, float16, int8)
            cache_dir: Directory to cache models
        """
        super().__init__(
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        # Map compute_type to appropriate PyTorch dtype
        self.dtype_map = {
            "float32": "float32",
            "float16": "float16",
            "int8": "int8",
        }
        
        # Initialize metrics
        self.metrics = {
            "load_time": 0,
            "inference_time": 0,
        }
    
    def load_model(self):
        """Load the OpenAI Whisper model."""
        if self.model is not None:
            return
            
        # Handle device mapping
        device = self.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Check for MPS (Apple Silicon)
            if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        
        # Map compute type
        dtype = self.dtype_map.get(self.compute_type, "float32")
        
        # Set download directory if provided
        if self.cache_dir:
            os.environ['WHISPER_DOWNLOAD_ROOT'] = self.cache_dir
        
        # Load the model
        self.model = whisper.load_model(
            self.model_size,
            device=device,
            download_root=self.cache_dir
        )
    
    def transcribe(
        self, 
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using OpenAI Whisper.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dictionary containing transcription results
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Set default options
        options = {
            "language": self.language,
            "verbose": False,
        }
        
        # Add prompt if provided
        if prompt:
            options["initial_prompt"] = prompt
            
        # Handle different audio input types
        if isinstance(audio, str):
            # For file paths
            start_time = time.time()
            result = self.model.transcribe(audio, **options)
            inference_time = time.time() - start_time
        else:
            # Fix NaN values in audio data that can cause exclamation mark outputs
            if np.issubdtype(audio.dtype, np.floating) and np.any(audio != audio):
                audio = audio.copy()  # Create a copy to avoid modifying the original
                audio[audio != audio] = 0.0  # Replace NaN values with zeros
                
            # For numpy arrays
            start_time = time.time()
            result = self.model.transcribe(audio, **options)  
            inference_time = time.time() - start_time
            
        # Filter out problematic exclamation marks in the main text
        text = result["text"]
        # Only filter if the text is mostly exclamation marks
        if text.strip():
            exclamation_count = text.count('!')
            if exclamation_count > 0 and exclamation_count / len(text.strip()) > 0.8:
                text = ""
        result["text"] = text
        
        # Filter out exclamation marks in segments
        if "segments" in result:
            for segment in result["segments"]:
                if "text" in segment:
                    segment_text = segment["text"]
                    # Only filter if the segment is mostly exclamation marks
                    if segment_text.strip():
                        exclamation_count = segment_text.count('!')
                        if exclamation_count > 0 and exclamation_count / len(segment_text.strip()) > 0.8:
                            segment_text = ""
                    segment["text"] = segment_text
            
        # Add inference time to the result
        result["inference_time"] = inference_time
        
        return result
