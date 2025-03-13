#!/usr/bin/env python
"""
Test script to validate the fix for NaN values causing exclamation marks in Whisper output.
"""

import numpy as np
import time
from whisper_term_asr.models.factory import create_model

def test_nan_handling():
    """Test that NaN values in audio are handled correctly."""
    print("Testing NaN handling in audio data...")
    
    # Create a model
    model_type = "lightning-mlx"  # The model type you were having issues with
    model_size = "distil-small.en"
    print(f"Creating {model_type} model ({model_size})...")
    
    model = create_model(model_type=model_type, model_size=model_size)
    
    try:
        # Create some test audio data (1 second of silence)
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)
        
        # Insert some NaN values which would previously cause exclamation marks
        nan_indices = np.random.choice(len(audio_data), size=int(len(audio_data) * 0.05), replace=False)
        audio_data[nan_indices] = np.nan
        
        print(f"Created test audio with {len(nan_indices)} NaN values")
        
        # Before our fix, this would return exclamation marks
        print("Transcribing audio with NaN values...")
        start_time = time.time()
        result = model.transcribe(audio_data)
        elapsed = time.time() - start_time
        
        # Check the results
        print(f"Transcription complete in {elapsed:.2f} seconds")
        print(f"Transcribed text: '{result['text']}'")
        
        # Check if we have any exclamation marks in the result
        if "!" in result["text"]:
            print("❌ Test FAILED: Exclamation marks found in the result!")
        else:
            print("✅ Test PASSED: No exclamation marks in the result")
            
        return not ("!" in result["text"])
        
    finally:
        # Clean up
        if model is not None:
            model.cleanup()

def main():
    """Run all tests."""
    print("=== Testing NaN handling fix ===")
    success = test_nan_handling()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
if __name__ == "__main__":
    main()
