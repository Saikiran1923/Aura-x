import html
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ProjectType(str, Enum):
    FASTAPI = "fastapi"
    FLASK = "flask"
    STATIC_HTML = "static_html"
    CLI = "cli"


@dataclass
class LaunchResult:
    project_type: ProjectType
    launched: bool
    browser_open_scheduled: bool
    details: str = ""


class ProjectLauncher:
    FLASK_URL = "http://127.0.0.1:5000"
    FASTAPI_URL = "http://127.0.0.1:8000"
    CLI_SUMMARY_FILE = "aura_x_summary.html"

    def __init__(self) -> None:
        self._background_processes: list[subprocess.Popen] = []

    def launch_project(self, project_root: Path) -> LaunchResult:
        resolved_root = project_root.resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            return LaunchResult(
                project_type=ProjectType.CLI,
                launched=False,
                browser_open_scheduled=False,
                details=f"Project path not found: {resolved_root}",
            )

        project_type, entry_path = self.detect_project_type(resolved_root)

        if project_type == ProjectType.FASTAPI:
            if entry_path is None:
                return LaunchResult(project_type, False, False, "FastAPI app.py not found.")
            return self._launch_fastapi(resolved_root, entry_path)

        if project_type == ProjectType.FLASK:
            if entry_path is None:
                return LaunchResult(project_type, False, False, "Flask app.py not found.")
            return self._launch_flask(resolved_root, entry_path)

        if project_type == ProjectType.STATIC_HTML:
            if entry_path is None:
                return LaunchResult(project_type, False, False, "index.html file not found.")
            self._open_browser_async(entry_path.resolve().as_uri(), delay_seconds=0.1)
            return LaunchResult(project_type, True, True, f"Opened {entry_path.name}")

        summary_file = self._generate_cli_summary_page(resolved_root)
        self._open_browser_async(summary_file.resolve().as_uri(), delay_seconds=0.1)
        return LaunchResult(project_type, True, True, f"Opened {summary_file.name}")

    def detect_project_type(self, project_root: Path) -> tuple[ProjectType, Path | None]:
        app_file = self._find_file_by_name(project_root, "app.py")
        if app_file is not None:
            app_content = self._safe_read_text(app_file)
            if self._is_fastapi_app(app_content):
                return ProjectType.FASTAPI, app_file
            if self._is_flask_app(app_content):
                return ProjectType.FLASK, app_file

        index_file = self._find_file_by_name(project_root, "index.html")
        if index_file is not None:
            return ProjectType.STATIC_HTML, index_file

        return ProjectType.CLI, None

    def _launch_flask(self, project_root: Path, app_file: Path) -> LaunchResult:
        command = [
            sys.executable,
            "-m",
            "flask",
            "--app",
            str(app_file),
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
        started = self._start_background_process(command=command, cwd=project_root)
        if not started:
            return LaunchResult(ProjectType.FLASK, False, False, "Failed to start Flask server.")
        self._open_browser_async(self.FLASK_URL, delay_seconds=1.5)
        return LaunchResult(ProjectType.FLASK, True, True, "Flask server started.")

    def _launch_fastapi(self, project_root: Path, app_file: Path) -> LaunchResult:
        module_ref = self._module_reference(project_root, app_file)
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module_ref}:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        started = self._start_background_process(command=command, cwd=project_root)
        if not started:
            return LaunchResult(ProjectType.FASTAPI, False, False, "Failed to start FastAPI server.")
        self._open_browser_async(self.FASTAPI_URL, delay_seconds=1.5)
        return LaunchResult(ProjectType.FASTAPI, True, True, "FastAPI server started.")

    def _module_reference(self, project_root: Path, app_file: Path) -> str:
        relative_module = app_file.resolve().relative_to(project_root.resolve()).with_suffix("")
        return ".".join(relative_module.parts)

    def _is_fastapi_app(self, content: str) -> bool:
        has_import = bool(re.search(r"from\s+fastapi\s+import\s+FastAPI", content))
        has_instance = bool(re.search(r"^\s*app\s*=\s*FastAPI\s*\(", content, flags=re.MULTILINE))
        return has_import and has_instance

    def _is_flask_app(self, content: str) -> bool:
        has_import = bool(re.search(r"from\s+flask\s+import\s+Flask", content))
        has_instance = bool(
            re.search(r"^\s*[A-Za-z_]\w*\s*=\s*Flask\s*\(", content, flags=re.MULTILINE)
        )
        return has_import and has_instance

    def _safe_read_text(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _find_file_by_name(self, root: Path, file_name: str) -> Path | None:
        direct_match = root / file_name
        if direct_match.exists() and direct_match.is_file():
            return direct_match

        lower_name = file_name.lower()
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if self._should_skip_path(candidate, root):
                continue
            if candidate.name.lower() == lower_name:
                return candidate
        return None

    def _should_skip_path(self, file_path: Path, root: Path) -> bool:
        try:
            relative_parts = file_path.resolve().relative_to(root.resolve()).parts
        except ValueError:
            return True
        skip_names = {"__pycache__", ".git", ".venv", "venv", "node_modules"}
        return any(part in skip_names for part in relative_parts)

    def _generate_cli_summary_page(self, project_root: Path) -> Path:
        summary_path = project_root / self.CLI_SUMMARY_FILE
        file_items = self._collect_file_list(project_root)
        list_markup = "\n".join(f"<li>{html.escape(item)}</li>" for item in file_items)

        page = (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"  <title>{html.escape(project_root.name)} - AURA-X Summary</title>\n"
            "  <style>\n"
            "    body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }\n"
            "    h1 { margin-bottom: 0.25rem; }\n"
            "    .muted { color: #555; margin-top: 0; }\n"
            "    ul { padding-left: 1.2rem; }\n"
            "    code { background: #f5f5f5; padding: 0.1rem 0.25rem; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            f"  <h1>{html.escape(project_root.name)}</h1>\n"
            "  <p class=\"muted\">CLI project generated by AURA-X.</p>\n"
            "  <p>Run your project from the terminal in this directory:</p>\n"
            f"  <p><code>{html.escape(str(project_root))}</code></p>\n"
            "  <h2>Generated files</h2>\n"
            f"  <ul>{list_markup}</ul>\n"
            "</body>\n"
            "</html>\n"
        )

        summary_path.write_text(page, encoding="utf-8")
        return summary_path

    def _collect_file_list(self, project_root: Path, limit: int = 200) -> list[str]:
        items: list[str] = []
        for file_path in sorted(project_root.rglob("*")):
            if not file_path.is_file():
                continue
            if self._should_skip_path(file_path, project_root):
                continue
            try:
                relative = file_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                continue
            items.append(relative.as_posix())
            if len(items) >= limit:
                items.append("... additional files omitted ...")
                break
        if not items:
            items.append("No files were generated.")
        return items

    def _start_background_process(self, command: list[str], cwd: Path) -> bool:
        popen_kwargs = self._background_popen_kwargs()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                **popen_kwargs,
            )
        except (OSError, ValueError):
            return False

        self._background_processes.append(process)
        time.sleep(0.8)
        return process.poll() is None

    def _background_popen_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            creationflags = 0
            creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
            if creationflags:
                kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _open_browser_async(self, target: str, delay_seconds: float = 0.0) -> None:
        thread = threading.Thread(
            target=self._open_browser_worker,
            args=(target, max(0.0, delay_seconds)),
            daemon=True,
            name="aura_x_browser_launcher",
        )
        thread.start()

    def _open_browser_worker(self, target: str, delay_seconds: float) -> None:
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            webbrowser.open_new_tab(target)
        except webbrowser.Error:
            return
