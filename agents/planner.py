import json
import subprocess
import re

MODEL_NAME = "qwen2.5:7b"
def call_llm(prompt: str) -> str:
    process = subprocess.Popen(
        ["ollama", "run", MODEL_NAME, "--format", "json"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    output, error = process.communicate(prompt)
    return output.strip()

def extract_json(text: str):
    """
    Extract first JSON object from LLM response.
    """
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None

def create_plan(user_request: str):
    structured_prompt = f"""
You are a senior software architect.

Break the following user request into a structured development plan.

Return ONLY valid JSON.
Do not explain.
Do not add markdown.
Do not add extra text.

Format exactly like this:

{{
  "project_name": "",
  "tech_stack": [],
  "tasks": [
    {{
      "step_number": 1,
      "description": "",
      "files_to_create": []
    }}
  ]
}}

User Request:
{user_request}
"""

    response = call_llm(structured_prompt)

    json_text = extract_json(response)

    if not json_text:
        print("No JSON found in LLM response.")
        print(response)
        return None

    try:
        plan = json.loads(json_text)
        return plan
    except json.JSONDecodeError:
        print("JSON still invalid.")
        print(json_text)
        return None
