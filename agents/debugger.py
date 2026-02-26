import asyncio
from typing import Any

from core.ollama_client import CPU_THREADS, KEEP_ALIVE, MODEL_NAME, OllamaClient

DEFAULT_OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.85,
    "num_ctx": 3072,
    "num_predict": 1000,
    "num_thread": CPU_THREADS,
}


class DebuggerAgent:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_client: OllamaClient | None = None,
        keep_alive: str = KEEP_ALIVE,
        ollama_options: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.ollama_client = ollama_client or OllamaClient()
        self.keep_alive = keep_alive
        self.ollama_options = ollama_options or DEFAULT_OLLAMA_OPTIONS.copy()

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
        return await asyncio.to_thread(
            self.ollama_client.generate,
            prompt,
            self.model_name,
            self.ollama_options,
            self.keep_alive,
        )

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
