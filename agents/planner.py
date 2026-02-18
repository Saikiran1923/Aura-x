import asyncio
import json
import os
import re
import time
from typing import Any

import requests

MODEL_NAME = "qwen2.5:7b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = int(os.getenv("AURAX_OLLAMA_TIMEOUT_SECONDS", "240"))
CONNECT_TIMEOUT_SECONDS = float(os.getenv("AURAX_OLLAMA_CONNECT_TIMEOUT_SECONDS", "10"))
MAX_RETRIES = int(os.getenv("AURAX_OLLAMA_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("AURAX_OLLAMA_RETRY_BACKOFF_SECONDS", "1.5"))
KEEP_ALIVE = "30m"
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)
DEFAULT_OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.8,
    "num_ctx": 2048,
    "num_predict": 250,
    "num_thread": CPU_THREADS,
}


class PlannerAgent:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_api_url: str = OLLAMA_API_URL,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        connect_timeout_seconds: float = CONNECT_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        keep_alive: str = KEEP_ALIVE,
        ollama_options: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.ollama_api_url = ollama_api_url
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.keep_alive = keep_alive
        self.ollama_options = ollama_options or DEFAULT_OLLAMA_OPTIONS.copy()
        self.session = requests.Session()

    async def create_plan(self, user_request: str) -> dict[str, Any]:
        cleaned_request = user_request.strip()
        if not cleaned_request:
            raise ValueError("User request cannot be empty.")

        prompt = self._build_prompt(cleaned_request)
        llm_response = await self._generate(prompt)
        parsed_json = self._safe_parse_json(llm_response)
        validated_plan = self._validate_plan(parsed_json)
        return validated_plan

    def _build_prompt(self, user_request: str) -> str:
        return (
            "You are Planner Agent for a local autonomous AI system.\n"
            "Create a strict concise development plan from the user request.\n"
            "Return only valid JSON.\n"
            "Do not include markdown, comments, or extra text.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "project_name": "string",\n'
            '  "tech_stack": ["string"],\n'
            '  "tasks": [\n'
            "    {\n"
            '      "step_number": 1,\n'
            '      "description": "string",\n'
            '      "files_to_create": ["path/filename.ext"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"User request:\n{user_request}"
        )

    async def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self.ollama_options,
        }
        return await asyncio.to_thread(self._post_to_ollama, payload)

    def _post_to_ollama(self, payload: dict[str, Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.ollama_api_url,
                    json=payload,
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise RuntimeError("Ollama API returned unexpected payload format.")

                ollama_error = body.get("error")
                if isinstance(ollama_error, str) and ollama_error.strip():
                    raise RuntimeError(f"Ollama API error: {ollama_error.strip()}")

                generated = body.get("response", "")
                if not isinstance(generated, str) or not generated.strip():
                    raise RuntimeError("Ollama API returned an empty response.")
                return generated.strip()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_seconds * (2**attempt)
                    if backoff:
                        time.sleep(backoff)
                    continue

                error_message = f"Ollama API request failed: {exc}"
                if isinstance(exc, requests.Timeout):
                    error_message = (
                        "Ollama API request timed out. "
                        f"Read timeout={self.timeout_seconds}s. "
                        "Set AURAX_OLLAMA_TIMEOUT_SECONDS to a higher value if needed."
                    )
                raise RuntimeError(error_message) from exc
            except ValueError as exc:
                raise RuntimeError("Ollama API returned non-JSON response.") from exc

        raise RuntimeError(f"Ollama API request failed: {last_error}")

    def _safe_parse_json(self, raw_text: str) -> dict[str, Any]:
        stripped = raw_text.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

        direct_result = self._try_load_json_object(stripped)
        if direct_result is not None:
            return direct_result

        decoder = json.JSONDecoder()
        index = stripped.find("{")
        while index != -1:
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                index = stripped.find("{", index + 1)
                continue
            if isinstance(candidate, dict):
                return candidate
            index = stripped.find("{", index + 1)

        raise RuntimeError("Planner output did not contain valid JSON object.")

    def _try_load_json_object(self, text: str) -> dict[str, Any] | None:
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(candidate, dict):
            return candidate
        return None

    def _validate_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        project_name = plan.get("project_name")
        if not isinstance(project_name, str) or not project_name.strip():
            raise RuntimeError("Planner output missing valid 'project_name'.")

        tasks = plan.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise RuntimeError("Planner output missing non-empty 'tasks' list.")

        normalized_tasks: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue

            description = task.get("description")
            files_to_create = task.get("files_to_create")
            if not isinstance(description, str) or not description.strip():
                continue
            if not isinstance(files_to_create, list):
                continue

            normalized_files = [
                str(file_path).strip()
                for file_path in files_to_create
                if str(file_path).strip()
            ]
            if not normalized_files:
                continue

            step_number = task.get("step_number")
            if not isinstance(step_number, int):
                step_number = index

            normalized_tasks.append(
                {
                    "step_number": step_number,
                    "description": description.strip(),
                    "files_to_create": normalized_files,
                }
            )

        if not normalized_tasks:
            raise RuntimeError("Planner output has no valid tasks after validation.")

        tech_stack = plan.get("tech_stack", [])
        if not isinstance(tech_stack, list):
            tech_stack = []

        normalized_tech_stack = [
            str(item).strip() for item in tech_stack if str(item).strip()
        ]

        return {
            "project_name": project_name.strip(),
            "tech_stack": normalized_tech_stack,
            "tasks": normalized_tasks,
        }


async def create_plan(user_request: str) -> dict[str, Any]:
    planner = PlannerAgent()
    return await planner.create_plan(user_request)
