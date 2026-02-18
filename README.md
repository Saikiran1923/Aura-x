# AURA-X

AURA-X is a local autonomous AI system that uses Ollama through its HTTP API to plan, generate, execute, and auto-correct Python projects.

## Architecture

- Planner Agent (`agents/planner.py`)
  - Converts a user request into strict JSON plan output.
- Coder Agent (`agents/coder.py`)
  - Generates file content and writes files safely into the project directory.
- Debugger Agent (`agents/debugger.py`)
  - Attempts one automated correction pass when execution errors occur.
- Execution Engine (`core/executor.py`)
  - Executes generated Python files asynchronously with timeout handling.

## Project Structure

```
aura-x/
├── agents/
│   ├── planner.py
│   ├── coder.py
│   └── debugger.py
├── core/
│   └── executor.py
├── projects/
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Ollama running locally
- Model available locally: `qwen2.5:7b`

## Setup

1. Install Python dependency:
   - `pip install -r requirements.txt`
2. Ensure Ollama is running:
   - `ollama serve`
3. Ensure model exists:
   - `ollama pull qwen2.5:7b`

## Run

- `python main.py`

Then enter your project request when prompted.

## Runtime Behavior

1. Accepts a user request.
2. Generates a strict JSON development plan.
3. Generates and writes files into `projects/<project_name>/`.
4. Executes generated `.py` files.
5. Detects execution errors.
6. Attempts one automatic correction and re-runs once.
