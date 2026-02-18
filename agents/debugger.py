import subprocess

MODEL_NAME = "deepseek-coder"

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

def fix_code(original_code: str, error_message: str):
    prompt = f"""
You are an expert Python debugger.

Fix the following code based on the error.

Error:
{error_message}

Code:
{original_code}

Return only corrected code.
"""
    return call_llm(prompt)
