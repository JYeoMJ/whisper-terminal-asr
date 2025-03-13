"""
Faster Whisper model implementation.
Uses the Faster Whisper implementation based on CTranslate2.
"""

import os
import time
import numpy as np
import re
import soundfile as sf
from typing import Dict, Any, Optional, Union, List

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

from .base import ASRModel


class FasterWhisperModel(ASRModel):
    """
    Implementation of the Faster Whisper model based on CTranslate2.
    Optimized for CPU inference with better performance than the original OpenAI version.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float32",
        cache_dir: Optional[str] = None,
        cpu_threads: int = 4,
        beam_size: int = 5,
    ):
        """
        Initialize the Faster Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'fr', None for auto)
            device: Device to run on ('cpu', 'cuda', None for auto)
            compute_type: Computation type/precision (float32, float16, int8)
            cache_dir: Directory to cache models
            cpu_threads: Number of CPU threads to use
            beam_size: Beam size for decoding
        """
        super().__init__(
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        
        # Map compute_type to appropriate CTranslate2 dtype
        self.dtype_map = {
            "float32": "float32",
            "float16": "float16",
            "int8": "int8",
        }
    
    def load_model(self):
        """Load the Faster Whisper model."""
        if self.model is not None:
            return
            
        # Handle device mapping
        device = self.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Map compute type
        compute_type = self.dtype_map.get(self.compute_type, "float32")
        
        # Load the model
        download_root = self.cache_dir if self.cache_dir else None
        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            cpu_threads=self.cpu_threads,
        )
    
    def transcribe(
        self, 
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Faster Whisper.
        
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
            # File path
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                initial_prompt=prompt,
                beam_size=self.beam_size,
            )
        else:
            # NumPy array
            # Fix NaN values in audio data that can cause exclamation mark outputs
            if hasattr(audio, "dtype") and np.issubdtype(audio.dtype, np.floating) and np.any(audio != audio):
                audio = audio.copy()  # Create a copy to avoid modifying the original
                audio[audio != audio] = 0.0  # Replace NaN values with zeros
                
            if hasattr(audio, "dtype") and np.issubdtype(audio.dtype, np.floating):
                # If audio is float and already normalized between -1 and 1, pass directly
                audio_data = audio
            else:
                # Convert to float32 and normalize if not already
                audio_data = audio.astype(np.float32) / np.iinfo(np.int16).max
                
            # Convert segments generator to list so we can reuse it
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                initial_prompt=prompt,
                beam_size=self.beam_size,
            )
        
        # Convert segments to list to make it serializable
        segments_list = []
        for segment in segments:
            # Filter exclamation marks in segment text
            segment_text = segment.text
            if segment_text:
                # Only filter if the segment is mostly exclamation marks
                if segment_text.strip():
                    exclamation_count = segment_text.count('!')
                    if exclamation_count > 0 and exclamation_count / len(segment_text.strip()) > 0.8:
                        segment_text = ""
            
            segments_list.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment_text,
                "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                         for word in (segment.words or [])],
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            })
            
        # Join the cleaned segment texts
        full_text = " ".join(segment["text"] for segment in segments_list if segment["text"])
        
        # Apply exclamation mark filtering to the full text as a final check
        if full_text:
            # Only filter if the text is mostly exclamation marks
            exclamation_count = full_text.count('!')
            if exclamation_count > 0 and exclamation_count / len(full_text.strip()) > 0.8:
                full_text = ""
        
        result = {
            "segments": segments_list,
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
        }
        
        return result
