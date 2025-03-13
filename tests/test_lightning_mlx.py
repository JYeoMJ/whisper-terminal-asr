"""
Tests for the Lightning Whisper MLX implementation.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from whisper_term_asr.models.whisper_lightning_mlx import WhisperLightningMLX
from whisper_term_asr.models.factory import create_model


class TestWhisperLightningMLX(unittest.TestCase):
    """Test cases for the WhisperLightningMLX class."""
    
    @patch('lightning_whisper_mlx.LightningWhisperMLX')
    def test_init(self, mock_lightning_mlx):
        """Test initialization of WhisperLightningMLX."""
        model = WhisperLightningMLX(
            model_size="distil-medium.en",
            batch_size=12,
            quantization=None,
            device="cpu",
            compute_type="float16",
            cache_dir=None
        )
        
        # Check if attributes were set correctly
        self.assertEqual(model.model_size, "distil-medium.en")
        self.assertEqual(model.batch_size, 12)
        self.assertIsNone(model.quantization)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.compute_type, "float16")
        self.assertIsNone(model.cache_dir)
        self.assertIsNone(model.model)

    @patch('lightning_whisper_mlx.LightningWhisperMLX')
    def test_load_model(self, mock_lightning_mlx):
        """Test loading the model."""
        # Create mock instance
        mock_instance = MagicMock()
        mock_lightning_mlx.return_value = mock_instance
        
        # Create model and load it
        model = WhisperLightningMLX(model_size="distil-small.en", batch_size=8)
        model.load_model()
        
        # Check if LightningWhisperMLX was instantiated with correct args
        mock_lightning_mlx.assert_called_once_with(
            model="distil-small.en", 
            batch_size=8,
            quant=None,
        )
        
        # Check if model was assigned
        self.assertEqual(model.model, mock_instance)

    @patch('lightning_whisper_mlx.LightningWhisperMLX')
    def test_load_model_with_quantization(self, mock_lightning_mlx):
        """Test loading model with quantization."""
        # Create mock instance
        mock_instance = MagicMock()
        mock_lightning_mlx.return_value = mock_instance
        
        # Create model with quantization and load it
        model = WhisperLightningMLX(model_size="distil-medium.en", batch_size=12, quantization="8bit")
        model.load_model()
        
        # Check if LightningWhisperMLX was instantiated with correct args
        mock_lightning_mlx.assert_called_once_with(
            model="distil-medium.en", 
            batch_size=12,
            quant="8bit",
        )
    
    @patch('lightning_whisper_mlx.LightningWhisperMLX')
    def test_transcribe_file(self, mock_lightning_mlx):
        """Test transcribe method with a file path."""
        # Create mock instance with transcribe method
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = {"text": "Hello world"}
        mock_lightning_mlx.return_value = mock_instance
        
        # Create model and load it
        model = WhisperLightningMLX(model_size="distil-small.en")
        model.load_model()
        
        # Test transcribe with a file path
        result = model.transcribe("test_audio.wav")
        
        # Check if transcribe was called and result is correct
        mock_instance.transcribe.assert_called_once_with("test_audio.wav")
        self.assertEqual(result, {"text": "Hello world"})

    @patch('lightning_whisper_mlx.LightningWhisperMLX')
    def test_transcribe_array(self, mock_lightning_mlx):
        """Test transcribe method with a numpy array."""
        # Create mock instance with transcribe method
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = {"text": "Hello world"}
        mock_lightning_mlx.return_value = mock_instance
        
        # Create model and load it
        model = WhisperLightningMLX(model_size="distil-small.en")
        model.load_model()
        
        # Test transcribe with a numpy array
        audio_array = np.zeros(16000)  # 1 second of silence at 16kHz
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Mock the save_audio function
            with patch('whisper_term_asr.models.whisper_lightning_mlx.save_audio') as mock_save_audio:
                mock_save_audio.return_value = temp_filename
                
                result = model.transcribe(audio_array)
                
                # Check if save_audio was called
                mock_save_audio.assert_called_once()
                
                # Check if transcribe was called with temp file and result is correct
                mock_instance.transcribe.assert_called_once_with(temp_filename)
                self.assertEqual(result, {"text": "Hello world"})
                
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    @patch('whisper_term_asr.models.factory.WhisperLightningMLX')
    def test_factory_integration(self, mock_lightning_mlx_class):
        """Test integration with the factory."""
        # Create a mock instance
        mock_instance = MagicMock()
        mock_lightning_mlx_class.return_value = mock_instance
        
        # Create model through factory
        model = create_model(
            model_type="lightning-mlx",
            model_size="distil-medium.en",
            batch_size=12,
            quantization="4bit"
        )
        
        # Check if correct class was instantiated with correct parameters
        mock_lightning_mlx_class.assert_called_once_with(
            model_size="distil-medium.en",
            device=None,
            compute_type="float16",
            language=None,
            cache_dir=None,
            batch_size=12,
            quantization="4bit"
        )
        
        # Check if returned model is our mock
        self.assertEqual(model, mock_instance)


if __name__ == "__main__":
    unittest.main()
