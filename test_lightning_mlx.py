#!/usr/bin/env python
"""
Test script for Lightning Whisper MLX integration.
"""

import os
import tempfile
import time
import numpy as np
import soundfile as sf
from rich.console import Console

# Import our implementation
from whisper_term_asr.models.whisper_lightning_mlx import LightningWhisperMLXModel

# Create a console for nice output
console = Console()

def create_test_audio(filepath, duration=3, sample_rate=16000):
    """
    Create a test audio file with a sine wave.
    """
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Save to file
    sf.write(filepath, audio, sample_rate)
    return filepath

def main():
    """Main test function."""
    console.print("[bold green]Testing Lightning Whisper MLX Implementation[/bold green]")
    
    # Create a temporary file for the test audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        test_audio_path = temp_file.name
    
    try:
        # Create test audio file
        console.print("[yellow]Creating test audio file...[/yellow]")
        create_test_audio(test_audio_path)
        console.print(f"[green]Test audio saved to: {test_audio_path}[/green]")
        
        # Initialize model
        console.print("[yellow]Initializing Lightning Whisper MLX model...[/yellow]")
        model = LightningWhisperMLXModel(
            model_size="distil-small.en",  # Use a smaller model for faster loading
            batch_size=8,
            quantization=None
        )
        
        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        start_time = time.time()
        model.load_model()
        load_time = time.time() - start_time
        console.print(f"[green]Model loaded in {load_time:.2f} seconds[/green]")
        
        # Transcribe audio
        console.print("[yellow]Transcribing audio...[/yellow]")
        start_time = time.time()
        result = model.transcribe(test_audio_path)
        inference_time = time.time() - start_time
        
        # Print results
        console.print(f"[bold green]Transcription completed in {inference_time:.2f} seconds[/bold green]")
        console.print(f"[bold]Result:[/bold] {result['text']}")
        
    finally:
        # Clean up
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            console.print("[yellow]Removed temporary audio file[/yellow]")

if __name__ == "__main__":
    main()
