from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


def load_env(env_path: Optional[str] = None) -> None:
    if env_path is None:
        env_path = str(Path(__file__).parent.parent / ".env")
    
    if not os.path.exists(env_path):
        logger.warning(f".env file not found at {env_path}")
        return
    
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMConfig:
    model: str = "meta-llama/llama-3.1-70b-instruct:free"
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: int = 30
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    enabled: bool = True


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


class OpenRouterClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        load_env()
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it in .env file or pass in config."
            )

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
            timeout=self.config.timeout_seconds,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.config.enabled:
            raise RuntimeError("LLM client is disabled via config")

        self._ensure_client()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                messages=messages,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("LLM client is disabled via config")

        self._ensure_client()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            tool_name = "role_classification"
            tool_description = "Classify the speaker role in an interview transcript"

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": response_schema,
                    },
                }
            ]

            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )

            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.tool_calls and len(message.tool_calls) > 0:
                    tool_call = message.tool_calls[0]
                    if tool_call.function and tool_call.function.arguments:
                        return json.loads(tool_call.function.arguments)

            logger.warning("No tool call found in OpenRouter response")
            return {}

        except Exception as e:
            logger.error(f"OpenRouter structured API call failed: {e}")
            raise


class MockLLMClient:
    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        self._responses = responses or []
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response.get("text", "")
        return ""

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response.get("structured", {})
        return {}

    def reset(self):
        self._call_count = 0
