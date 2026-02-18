import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECTS_ROOT = Path("projects")
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 45


@dataclass
class ExecutionResult:
    file_path: Path
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def has_error(self) -> bool:
        return self.timed_out or self.return_code != 0 or bool(self.stderr.strip())


class ExecutionEngine:
    def __init__(
        self,
        projects_root: Path = PROJECTS_ROOT,
        execution_timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        self.projects_root = projects_root
        self.execution_timeout_seconds = execution_timeout_seconds
        self.python_executable = sys.executable

    async def run_python_file(self, project_name: str, file_name: str) -> ExecutionResult:
        file_path = self._resolve_project_path(project_name, file_name)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot execute missing file: {file_path}")
        if file_path.suffix.lower() != ".py":
            raise ValueError(f"ExecutionEngine only supports Python files: {file_path}")

        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(file_path),
            cwd=str(file_path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.execution_timeout_seconds,
            )
            timed_out = False
            return_code = process.returncode if process.returncode is not None else 1
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True
            return_code = 124

        stdout = stdout_bytes.decode("utf-8", errors="ignore") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="ignore") if stderr_bytes else ""

        if timed_out:
            timeout_message = (
                f"Execution timed out after {self.execution_timeout_seconds} seconds."
            )
            stderr = f"{stderr}\n{timeout_message}".strip()

        return ExecutionResult(
            file_path=file_path,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def _resolve_project_path(self, project_name: str, file_name: str) -> Path:
        project_root = (self.projects_root / project_name.strip()).resolve()
        target_path = (project_root / file_name.strip()).resolve()
        if project_root not in target_path.parents and target_path != project_root:
            raise ValueError(f"Invalid path outside project root: {file_name}")
        return target_path


async def run_python_file(project_name: str, file_name: str) -> ExecutionResult:
    executor = ExecutionEngine()
    return await executor.run_python_file(project_name, file_name)
