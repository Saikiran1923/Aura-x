import asyncio
import json
import re
from pathlib import Path

from agents.coder import CoderAgent
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from core.executor import ExecutionEngine, ExecutionResult


class AuraXSystem:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.debugger = DebuggerAgent()
        self.executor = ExecutionEngine()

    async def run(self, user_request: str) -> int:
        try:
            plan = await self.planner.create_plan(user_request)
        except Exception as exc:
            print(f"Planning failed: {exc}")
            return 1

        project_name = self._sanitize_project_name(plan["project_name"])
        plan["project_name"] = project_name

        print("\nGenerated Plan (strict JSON):")
        print(json.dumps(plan, indent=2))

        project_root = Path("projects") / project_name
        await asyncio.to_thread(project_root.mkdir, parents=True, exist_ok=True)

        for task in plan["tasks"]:
            description = task["description"]
            for file_name in task["files_to_create"]:
                print(f"\nGenerating {file_name} ...")
                try:
                    code = await self.coder.generate_file_code(
                        file_name=file_name,
                        task_description=description,
                        project_request=None,
                    )
                except Exception as exc:
                    print(f"Code generation failed for {file_name}: {exc}")
                    continue

                try:
                    written_path = await self.coder.write_file(
                        project_name=project_name,
                        file_name=file_name,
                        content=code,
                    )
                    print(f"Created: {written_path}")
                except Exception as exc:
                    print(f"Write failed for {file_name}: {exc}")
                    continue

                if written_path.suffix.lower() != ".py":
                    continue

                result = await self._execute_file(project_name, file_name)
                if not result.has_error:
                    continue

                print("Error detected. Attempting automatic correction (1 attempt).")
                await self._attempt_fix_once(
                    project_name=project_name,
                    file_name=file_name,
                    first_result=result,
                )

        print("\nProject generation flow completed.")
        print(f"Output directory: {project_root.resolve()}")
        return 0

    async def _execute_file(self, project_name: str, file_name: str) -> ExecutionResult:
        try:
            result = await self.executor.run_python_file(project_name, file_name)
        except Exception as exc:
            fake_path = (Path("projects") / project_name / file_name).resolve()
            result = ExecutionResult(
                file_path=fake_path,
                return_code=1,
                stdout="",
                stderr=str(exc),
                timed_out=False,
            )

        self._print_execution_result(result)
        return result

    async def _attempt_fix_once(
        self,
        project_name: str,
        file_name: str,
        first_result: ExecutionResult,
    ) -> None:
        target_path = (Path("projects") / project_name / file_name).resolve()
        try:
            original_code = await asyncio.to_thread(target_path.read_text, "utf-8")
        except Exception as exc:
            print(f"Cannot read file for debugging ({file_name}): {exc}")
            return

        try:
            fixed_code = await self.debugger.fix_code(
                original_code=original_code,
                error_message=first_result.stderr or f"Exit code: {first_result.return_code}",
                file_name=file_name,
            )
        except Exception as exc:
            print(f"Debugger failed for {file_name}: {exc}")
            return

        try:
            await self.coder.write_file(
                project_name=project_name,
                file_name=file_name,
                content=fixed_code,
            )
        except Exception as exc:
            print(f"Failed to write corrected code for {file_name}: {exc}")
            return

        print("Re-running file after auto-correction...")
        second_result = await self._execute_file(project_name, file_name)
        if second_result.has_error:
            print("Automatic correction failed.")
        else:
            print("Automatic correction successful.")

    def _sanitize_project_name(self, project_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", project_name.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "generated_project"

    def _print_execution_result(self, result: ExecutionResult) -> None:
        print(f"\nRunning {result.file_path.name} ...")
        print(f"Return code: {result.return_code}")
        if result.stdout.strip():
            print("STDOUT:")
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr.rstrip())


async def async_main() -> int:
    user_input = (await asyncio.to_thread(input, "Enter project request: ")).strip()
    if not user_input:
        print("Request cannot be empty.")
        return 1

    system = AuraXSystem()
    return await system.run(user_input)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
