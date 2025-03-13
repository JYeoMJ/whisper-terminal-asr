"""
Base model interface for ASR models.
Defines the common API for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import time

class ASRModel(ABC):
    """Abstract base class for all ASR model implementations."""
    
    AVAILABLE_SIZES = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float32",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the ASR model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'fr', None for auto)
            device: Device to run on ('cpu', 'cuda', 'mps', None for auto)
            compute_type: Computation type/precision (float32, float16, int8)
            cache_dir: Directory to cache models
        """
        if model_size not in self.AVAILABLE_SIZES:
            raise ValueError(f"Model size must be one of {self.AVAILABLE_SIZES}")
            
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = cache_dir
        self.model = None
    
    @abstractmethod
    def load_model(self):
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def transcribe(
        self, 
        audio: Union[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dictionary containing transcription results
        """
        pass
    
    def benchmark(
        self, 
        audio: Union[str, np.ndarray],
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Benchmark the model on an audio sample.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            num_runs: Number of runs to average over
            
        Returns:
            Dictionary with benchmark results
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
            
        load_times = []
        inference_times = []
        
        for i in range(num_runs):
            # Measure loading time (if applicable)
            start_time = time.time()
            # Some implementations may need to reload parts of the model
            load_time = time.time() - start_time
            load_times.append(load_time)
            
            # Measure inference time
            start_time = time.time()
            result = self.transcribe(audio)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
        return {
            "model_type": self.__class__.__name__,
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "avg_load_time": sum(load_times) / num_runs,
            "avg_inference_time": sum(inference_times) / num_runs,
            "min_inference_time": min(inference_times),
            "max_inference_time": max(inference_times),
            "std_inference_time": np.std(inference_times) if len(inference_times) > 1 else 0,
            "num_runs": num_runs,
            "result": result,
        }

    def cleanup(self):
        """
        Release resources held by the model.
        
        This method should be called when the model is no longer needed
        to ensure proper resource cleanup and prevent memory leaks.
        """
        # Set model to None to allow for garbage collection
        # This doesn't delete downloaded weights/files
        self.model = None
        
        # Force garbage collection
        import gc
        gc.collect()
