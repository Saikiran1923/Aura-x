import os
import time
from typing import Any

import requests

MODEL_NAME = "qwen2.5:7b"
OLLAMA_BASE_URL = os.getenv("AURAX_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

REQUEST_TIMEOUT_SECONDS = int(os.getenv("AURAX_OLLAMA_TIMEOUT_SECONDS", "240"))
CONNECT_TIMEOUT_SECONDS = float(os.getenv("AURAX_OLLAMA_CONNECT_TIMEOUT_SECONDS", "10"))
MAX_RETRIES = int(os.getenv("AURAX_OLLAMA_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("AURAX_OLLAMA_RETRY_BACKOFF_SECONDS", "1.5"))
KEEP_ALIVE = os.getenv("AURAX_OLLAMA_KEEP_ALIVE", "30m")
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        connect_timeout_seconds: float = CONNECT_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
    ) -> None:
        base = base_url.rstrip("/")
        self.generate_url = f"{base}/api/generate"
        self.tags_url = f"{base}/api/tags"
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        model_name: str = MODEL_NAME,
        options: dict[str, Any] | None = None,
        keep_alive: str = KEEP_ALIVE,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options

        body = self._request_json("POST", self.generate_url, json_payload=payload)
        error_text = body.get("error")
        if isinstance(error_text, str) and error_text.strip():
            raise RuntimeError(f"Ollama API error: {error_text.strip()}")

        generated = body.get("response", "")
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError("Ollama API returned an empty response.")
        return generated.strip()

    def check_server(self) -> tuple[bool, str]:
        try:
            self._request_json("GET", self.tags_url, retry_count=0)
        except Exception as exc:
            return False, f"Ollama server unavailable: {exc}"
        return True, "Ollama server is reachable."

    def check_model(self, model_name: str = MODEL_NAME) -> tuple[bool, str]:
        try:
            body = self._request_json("GET", self.tags_url, retry_count=0)
        except Exception as exc:
            return False, f"Failed to query local models: {exc}"

        models = body.get("models", [])
        if not isinstance(models, list):
            return False, "Unexpected model list response from Ollama."

        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            model = str(item.get("model", "")).strip()
            if name == model_name or model == model_name:
                return True, f"Model '{model_name}' is available."

        return False, f"Model '{model_name}' not found. Run: ollama pull {model_name}"

    def _request_json(
        self,
        method: str,
        url: str,
        json_payload: dict[str, Any] | None = None,
        retry_count: int | None = None,
    ) -> dict[str, Any]:
        retries = self.max_retries if retry_count is None else max(0, retry_count)
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise RuntimeError("Ollama API returned unexpected payload format.")
                return body
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries:
                    backoff = self.retry_backoff_seconds * (2**attempt)
                    if backoff:
                        time.sleep(backoff)
                    continue

                if isinstance(exc, requests.Timeout):
                    raise RuntimeError(
                        "Ollama API request timed out. "
                        f"Read timeout={self.timeout_seconds}s. "
                        "Increase AURAX_OLLAMA_TIMEOUT_SECONDS if needed."
                    ) from exc
                raise RuntimeError(f"Ollama API request failed: {exc}") from exc
            except ValueError as exc:
                raise RuntimeError("Ollama API returned non-JSON response.") from exc

        raise RuntimeError(f"Ollama API request failed: {last_error}")
