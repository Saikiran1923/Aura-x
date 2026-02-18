import subprocess
import os

def run_python_file(project_name: str, file_name: str):
    file_path = os.path.join("projects", project_name, file_name)

    result = subprocess.run(
        ["python", file_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout, result.stderr
