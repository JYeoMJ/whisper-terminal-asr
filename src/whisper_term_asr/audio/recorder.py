"""
Audio recording functionality for the Whisper Terminal ASR application.
Provides both continuous streaming and manual recording capabilities.
"""

import os
import time
import threading
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, Callable, Union, List, Dict, Any

class AudioRecorder:
    """
    Audio recorder class for capturing microphone input.
    Supports both continuous streaming and manual recording modes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        block_duration: float = 0.5,
        device: Optional[Union[str, int]] = None,
        output_dir: str = "recordings",
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            dtype: Audio data type
            block_duration: Duration of each audio block in seconds
            device: Audio device to use (None for default)
            output_dir: Directory to save recordings
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.block_duration = block_duration
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # State variables
        self.recording = False
        self.streaming = False
        self.stream = None
        self.audio_buffer = []
        self.callback_fn = None
        self._lock = threading.Lock()
        
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio devices."""
        return sd.query_devices()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for the audio stream."""
        if status:
            print(f"Status: {status}")
        
        with self._lock:
            self.audio_buffer.append(indata.copy())
            
            if self.callback_fn is not None:
                # Convert buffer to numpy array for processing
                audio_data = np.concatenate(self.audio_buffer)
                self.callback_fn(audio_data, self.sample_rate)
    
    def start_streaming(self, callback_fn: Callable[[np.ndarray, int], None] = None):
        """
        Start continuous audio streaming.
        
        Args:
            callback_fn: Function to call with audio data (audio_data, sample_rate)
        """
        if self.streaming:
            print("Already streaming")
            return
            
        self.streaming = True
        self.callback_fn = callback_fn
        self.audio_buffer = []
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * self.block_duration),
            dtype=self.dtype,
            device=self.device
        )
        
        self.stream.start()
        print(f"Started streaming from device: {self.stream.device}")
    
    def stop_streaming(self) -> Optional[np.ndarray]:
        """
        Stop continuous audio streaming.
        
        Returns:
            The full recorded audio buffer as a numpy array, or None if empty
        """
        if not self.streaming:
            print("Not streaming")
            return None
            
        self.stream.stop()
        self.stream.close()
        self.streaming = False
        
        with self._lock:
            if not self.audio_buffer:
                return None
                
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            return audio_data
    
    def start_recording(self):
        """Start manual recording session."""
        if self.recording:
            print("Already recording")
            return
            
        self.recording = True
        self.audio_buffer = []
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * self.block_duration),
            dtype=self.dtype,
            device=self.device
        )
        
        self.stream.start()
        print(f"Recording started on device: {self.stream.device}")
    
    def stop_recording(self, save: bool = True) -> Optional[np.ndarray]:
        """
        Stop manual recording session.
        
        Args:
            save: Whether to save the recording to disk
            
        Returns:
            The recorded audio as a numpy array, or None if empty
        """
        if not self.recording:
            print("Not recording")
            return None
            
        self.stream.stop()
        self.stream.close()
        self.recording = False
        
        with self._lock:
            if not self.audio_buffer:
                return None
                
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            
            if save:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(self.output_dir, f"recording-{timestamp}.wav")
                sf.write(filename, audio_data, self.sample_rate)
                print(f"Recording saved to {filename}")
                
            return audio_data
    
    def record_fixed_duration(self, duration: float, save: bool = True) -> np.ndarray:
        """
        Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds
            save: Whether to save the recording to disk
            
        Returns:
            The recorded audio as a numpy array
        """
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device
        )
        sd.wait()
        
        if save:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.output_dir, f"recording-{timestamp}.wav")
            sf.write(filename, audio_data, self.sample_rate)
            print(f"Recording saved to {filename}")
            
        return audio_data
