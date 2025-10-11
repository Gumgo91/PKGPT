"""
Gemini API Client for NONMEM optimization
Supports multiple Gemini models for iterative improvement
"""

import os
import google.generativeai as genai
from typing import Dict, List, Optional
import time


class GeminiClient:
    """Client for interacting with Google Gemini API"""

    # Model configurations
    MODELS = {
        'flash': 'gemini-flash-latest',
        'flash-lite': 'gemini-flash-lite-latest',
        'pro': 'gemini-2.5-pro'
    }

    def __init__(self, api_key: Optional[str] = None, model_type: str = 'pro'):
        """
        Initialize Gemini client

        Args:
            api_key: Google Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_type: Model type to use ('flash', 'flash-lite', 'pro')
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        if model_type not in self.MODELS:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Valid options: {list(self.MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_name = self.MODELS[model_type]

        # Configure API
        genai.configure(api_key=self.api_key)

        # Initialize model with generation config
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        print(f"[OK] Initialized Gemini client with model: {self.model_name}")

    def generate(self, prompt: str, retry_attempts: int = 3) -> str:
        """
        Generate response from Gemini API with retry logic

        Args:
            prompt: Input prompt
            retry_attempts: Number of retry attempts on failure

        Returns:
            Generated text response
        """
        for attempt in range(retry_attempts):
            try:
                response = self.model.generate_content(prompt)

                # Check if response has valid text
                if response.text:
                    return response.text
                else:
                    print(f"[WARNING] Empty response received (attempt {attempt + 1}/{retry_attempts})")

            except Exception as e:
                print(f"[WARNING] Error generating response (attempt {attempt + 1}/{retry_attempts}): {e}")

                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate response after {retry_attempts} attempts: {e}")

        raise Exception("Failed to generate valid response")

    def switch_model(self, model_type: str):
        """
        Switch to a different Gemini model

        Args:
            model_type: New model type ('flash', 'flash-lite', 'pro')
        """
        if model_type not in self.MODELS:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Valid options: {list(self.MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_name = self.MODELS[model_type]

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        print(f"[OK] Switched to model: {self.model_name}")

    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.model_name


class MultiModelGeminiClient:
    """Client that can use multiple Gemini models in rotation"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize multi-model client

        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.clients = {}

        # Initialize all available models
        for model_type in GeminiClient.MODELS.keys():
            try:
                self.clients[model_type] = GeminiClient(self.api_key, model_type)
            except Exception as e:
                print(f"[WARNING] Failed to initialize {model_type} model: {e}")

        if not self.clients:
            raise Exception("No Gemini models could be initialized")

        self.current_model_type = 'pro'  # Default to pro model

    def generate(self, prompt: str, model_type: Optional[str] = None) -> str:
        """
        Generate response using specified or current model

        Args:
            prompt: Input prompt
            model_type: Model type to use (if None, uses current model)

        Returns:
            Generated text response
        """
        model_type = model_type or self.current_model_type

        if model_type not in self.clients:
            raise ValueError(f"Model type {model_type} not available")

        return self.clients[model_type].generate(prompt)

    def rotate_model(self) -> str:
        """
        Rotate to next available model

        Returns:
            New model type
        """
        available_models = list(self.clients.keys())
        current_idx = available_models.index(self.current_model_type)
        next_idx = (current_idx + 1) % len(available_models)
        self.current_model_type = available_models[next_idx]

        print(f"[ROTATE] Rotated to model: {self.clients[self.current_model_type].get_current_model()}")
        return self.current_model_type

    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return list(self.clients.keys())
