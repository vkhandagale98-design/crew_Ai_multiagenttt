# CrewAI Multi-Agent System for Project Development

Six CrewAI agents work in sequence: Requirements Analysis → Solution Architecture → Code Generation → Testing → Documentation → Project Report. The Streamlit app accepts project title, description, language, database, project type, users, complexity and features. Groq provides the LLM.

## Requirements
Use Python 3.11 or 3.12 (recommended for compatibility with AI packages).

## Windows setup (PowerShell)
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Groq configuration
Copy `.env.example` to `.env`, then set your actual key:
```env
GROQ_API_KEY=your_actual_key
GROQ_API_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.2
```
Keep `.env` private and never upload it to GitHub. Revoke any API key that has been exposed.

## Run
```powershell
streamlit run app.py
```
Open the local URL printed in the terminal (normally http://localhost:8501).

## Included Library Management prototype
```powershell
python prototype/library_app.py
pytest -q
```
The SQLite module includes add/view books, register students, issue books and return books. Tests use a temporary database.

## Outputs
Agent results are saved as six Markdown files in `outputs/`: requirements, architecture, prototype code, testing, documentation and project report.

## Project structure
```text
app.py
crew_service.py
requirements.txt
.env.example
README.md
prototype/library_app.py
tests/test_library.py
outputs/
```

## Troubleshooting
- Missing key: ensure `.env` is beside `app.py`, then restart Streamlit.
- Authentication/model error: verify your key and choose a model currently available to your Groq account.
- Import error: activate `venv` and reinstall `requirements.txt`.
- Python compatibility: use Python 3.11/3.12 if your installed version is too new for a dependency.

## Safety
The six agents save code for review; the app does not execute AI-generated code. Inspect generated code and dependencies before use. Use an isolated environment/container for untrusted code. The included library prototype is a deterministic example; do not treat unexecuted agent-generated code as tested.
