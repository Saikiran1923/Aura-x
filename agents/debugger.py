import asyncio
from typing import Any

import requests

MODEL_NAME = "qwen2.5:7b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 180


class DebuggerAgent:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_api_url: str = OLLAMA_API_URL,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.ollama_api_url = ollama_api_url
        self.timeout_seconds = timeout_seconds

    async def fix_code(
        self,
        original_code: str,
        error_message: str,
        file_name: str,
    ) -> str:
        if not original_code.strip():
            raise ValueError("original_code cannot be empty.")

        prompt = self._build_prompt(
            original_code=original_code,
            error_message=error_message.strip(),
            file_name=file_name.strip(),
        )
        raw_response = await self._generate(prompt)
        cleaned = self._clean_code_output(raw_response)
        if not cleaned:
            raise RuntimeError("Debugger agent returned empty corrected code.")
        return cleaned

    def _build_prompt(
        self,
        original_code: str,
        error_message: str,
        file_name: str,
    ) -> str:
        return (
            "You are Debugger Agent in a local autonomous AI system.\n"
            "Fix the provided file according to the runtime error.\n"
            "Return only corrected full file content.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n\n"
            f"Target file: {file_name}\n"
            f"Error output:\n{error_message}\n\n"
            f"Original file content:\n{original_code}\n"
        )

    async def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        return await asyncio.to_thread(self._post_to_ollama, payload)

    def _post_to_ollama(self, payload: dict[str, Any]) -> str:
        try:
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama API request failed: {exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama API returned non-JSON response.") from exc

        generated = body.get("response", "")
        if not isinstance(generated, str):
            raise RuntimeError("Ollama API payload missing response text.")
        return generated.strip()

    def _clean_code_output(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned


async def fix_code(original_code: str, error_message: str, file_name: str) -> str:
    debugger = DebuggerAgent()
    return await debugger.fix_code(original_code, error_message, file_name)
