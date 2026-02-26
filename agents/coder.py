import asyncio
from pathlib import Path
from typing import Any

from core.ollama_client import CPU_THREADS, KEEP_ALIVE, MODEL_NAME, OllamaClient

DEFAULT_OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.85,
    "num_ctx": 3072,
    "num_thread": CPU_THREADS,
}
PROJECTS_ROOT = Path("projects")


class CoderAgent:
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

    async def generate_file_code(
        self,
        file_name: str,
        task_description: str,
        project_request: str | None = None,
    ) -> str:
        cleaned_file_name = file_name.strip()
        if not cleaned_file_name:
            raise ValueError("file_name cannot be empty.")

        prompt = self._build_prompt(
            file_name=cleaned_file_name,
            task_description=task_description.strip(),
            project_request=(project_request or "").strip(),
        )
        raw_response = await self._generate(
            prompt=prompt,
            file_name=cleaned_file_name,
        )
        cleaned_content = self._clean_code_output(raw_response)
        if not cleaned_content:
            raise RuntimeError(f"Coder agent returned empty content for '{file_name}'.")
        return cleaned_content

    async def write_file(
        self,
        project_name: str,
        file_name: str,
        content: str,
    ) -> Path:
        project_path = PROJECTS_ROOT / project_name.strip()
        target_path = self._resolve_project_path(project_path, file_name.strip())
        await asyncio.to_thread(self._write_text_file, target_path, content)
        return target_path

    def _build_prompt(
        self,
        file_name: str,
        task_description: str,
        project_request: str,
    ) -> str:
        return (
            "You are Coder Agent in a local autonomous AI system.\n"
            "Generate production-ready file content.\n"
            "Return only raw file content.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n\n"
            f"Target file: {file_name}\n"
            f"Task description: {task_description}\n"
            f"Original user request: {project_request[:800]}\n"
        )

    async def _generate(self, prompt: str, file_name: str) -> str:
        options = self._build_options(file_name)
        return await asyncio.to_thread(
            self.ollama_client.generate,
            prompt,
            self.model_name,
            options,
            self.keep_alive,
        )

    def _build_options(self, file_name: str) -> dict[str, Any]:
        options = self.ollama_options.copy()
        lower_name = file_name.lower()
        if lower_name.endswith(".py"):
            options["num_predict"] = 900
        elif lower_name.endswith((".md", ".txt", ".json", ".yaml", ".yml")):
            options["num_predict"] = 650
        else:
            options["num_predict"] = 750
        return options

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

    def _resolve_project_path(self, project_path: Path, file_name: str) -> Path:
        project_root = project_path.resolve()
        target_path = (project_root / file_name).resolve()
        if project_root not in target_path.parents and target_path != project_root:
            raise ValueError(f"Invalid file path outside project root: {file_name}")
        return target_path

    def _write_text_file(self, file_path: Path, content: str) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


async def generate_file_code(
    file_name: str,
    description: str,
    project_request: str | None = None,
) -> str:
    coder = CoderAgent()
    return await coder.generate_file_code(file_name, description, project_request)


async def write_file(project_name: str, file_name: str, content: str) -> Path:
    coder = CoderAgent()
    return await coder.write_file(project_name, file_name, content)
