import subprocess
import os

MODEL_NAME = "qwen2.5:7b"
def call_llm(prompt: str) -> str:
    process = subprocess.Popen(
        ["ollama", "run", MODEL_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    output, error = process.communicate(prompt)
    return output.strip()

def generate_file_code(file_name: str, description: str):
    prompt = f"""
You are a senior Python developer.

Generate complete production-ready code for file: {file_name}

Description:
{description}

Return only the code.
"""
    return call_llm(prompt)

def write_file(project_name: str, file_name: str, content: str):
    project_path = os.path.join("projects", project_name)
    os.makedirs(project_path, exist_ok=True)

    file_path = os.path.join(project_path, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created: {file_path}")
