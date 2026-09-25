"""Llama 3.1 LLM service with support for Groq, Ollama, HuggingFace, OpenAI-compatible, and offline mock."""

from abc import ABC, abstractmethod
from typing import Optional
import requests
from config.settings import Settings, get_settings
from utils.logging_utils import get_logger

logger = get_logger("llama_service")


class BaseLLMService(ABC):
    """Abstract base class for Llama 3.1 LLM generation."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from LLM given prompt and optional system prompt."""
        pass


class GroqLlamaService(BaseLLMService):
    """Llama 3.1 inference via Groq Cloud API."""

    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 1024):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq provider.")
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(f"Calling Groq with model '{self.model_name}'...")
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""


class OllamaLlamaService(BaseLLMService):
    """Llama 3.1 inference via local Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1", temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        logger.info(f"Calling Ollama at {url} with model '{self.model_name}'...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


class HuggingFaceLlamaService(BaseLLMService):
    """Llama 3.1 inference via Hugging Face Serverless / Inference Endpoints."""

    def __init__(self, api_token: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: float = 0.2, max_tokens: int = 1024):
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN is required for HuggingFace provider.")
        self.api_token = api_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt or ''}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": max(0.01, self.temperature),
                "return_full_text": False,
            },
        }

        logger.info(f"Calling Hugging Face Inference API for {self.model_name}...")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        return str(data)


class OpenAICompatibleLlamaService(BaseLLMService):
    """Llama 3.1 inference via any OpenAI-compatible endpoint (Together, OpenRouter, vLLM, DeepInfra)."""

    def __init__(self, api_key: str, base_url: str, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", temperature: float = 0.2, max_tokens: int = 1024):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai_compatible provider.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        logger.info(f"Calling OpenAI-compatible endpoint at {url} with model '{self.model_name}'...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""


class MockLlamaService(BaseLLMService):
    """Deterministic offline fallback LLM service for testing and demonstrations without active API keys."""

    def __init__(self, model_name: str = "mock-llama-3.1"):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        logger.info("Executing MockLlamaService generation (offline/fallback mode)...")
        # Synthesize an answer directly highlighting the provided context
        lines = prompt.splitlines()
        context_lines = [l for l in lines if l.startswith("[") or "Context:" in l]

        return (
            "*(Running in Demo/Mock Mode - Configure GROQ_API_KEY or other provider in `.env` for live Llama 3.1 inference)*\n\n"
            "Based on the retrieved video transcript segments, here is the relevant information:\n\n"
            "The speaker discusses these points directly within the video segments provided. "
            "Please refer to the source timestamps below to view the exact discussion in YouTube."
        )


def get_llm_service(settings: Optional[Settings] = None) -> BaseLLMService:
    """Factory to instantiate the appropriate Llama 3.1 LLM service based on configuration."""
    cfg = settings or get_settings()
    provider = cfg.llm_provider.lower().strip()

    logger.info(f"Selecting LLM provider: {provider}")

    if provider == "groq":
        if cfg.groq_api_key:
            return GroqLlamaService(
                api_key=cfg.groq_api_key,
                model_name=cfg.model_name or "llama-3.1-8b-instant",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            logger.warning("GROQ_API_KEY is not set. Falling back to MockLlamaService for demonstration.")
            return MockLlamaService()

    elif provider == "ollama":
        return OllamaLlamaService(
            base_url=cfg.ollama_base_url,
            model_name=cfg.model_name or "llama3.1",
            temperature=cfg.temperature,
        )

    elif provider == "huggingface":
        if cfg.huggingface_api_token:
            return HuggingFaceLlamaService(
                api_token=cfg.huggingface_api_token,
                model_name=cfg.model_name or "meta-llama/Llama-3.1-8B-Instruct",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            logger.warning("HUGGINGFACE_API_TOKEN is not set. Falling back to MockLlamaService.")
            return MockLlamaService()

    elif provider == "openai_compatible":
        if cfg.openai_api_key:
            return OpenAICompatibleLlamaService(
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
                model_name=cfg.model_name or "meta-llama/Meta-Llama-3.1-8B-Instruct",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            logger.warning("OPENAI_API_KEY is not set. Falling back to MockLlamaService.")
            return MockLlamaService()

    elif provider == "mock":
        return MockLlamaService(model_name=cfg.model_name)

    else:
        logger.warning(f"Unknown LLM provider '{provider}'. Using MockLlamaService.")
        return MockLlamaService()
