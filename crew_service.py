import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv

load_dotenv()

def check_groq_configuration():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return False, "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
    return True, "Groq API key configured."

def make_llm():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY missing. Configure the .env file.")
    model = os.getenv("GROQ_API_MODEL", "openai/gpt-oss-120b").strip()
    return LLM(model=f"groq/{model}", api_key=key, temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")))

def run_project_crew(project, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = make_llm()
    brief = f"""TITLE: {project['title']}
DESCRIPTION: {project['description']}
LANGUAGE: {project['language']} | DATABASE: {project['database']}
TYPE: {project['project_type']} | USERS: {project['target_users']}
COMPLEXITY: {project['complexity']}
FEATURES: {', '.join(project['features'])}
CONSTRAINTS: {project['constraints']}"""

    def agent(role, goal, backstory):
        return Agent(role=role, goal=goal, backstory=backstory, llm=llm,
                     verbose=True, allow_delegation=False)

    req = agent("Requirements Analysis Agent", "Define complete, testable functional and non-functional requirements.",
                "You are a senior business analyst who writes traceable requirements.")
    arch = agent("Solution Architect Agent", "Design modules, architecture, data model and interfaces.",
                 "You are a practical software architect.")
    dev = agent("Code Generation Agent", "Produce a small coherent prototype and clearly labeled source files.",
                "You are a careful developer who writes maintainable code.")
    qa = agent("Testing Agent", "Create a traceable test strategy, test cases and automated test examples.",
               "You are a QA engineer focused on edge cases and reproducibility.")
    docs = agent("Documentation Agent", "Write accurate installation, developer and user documentation.",
                 "You are a technical writer.")
    report = agent("Project Report Agent", "Prepare a formal academic project report based on previous outputs.",
                   "You are an academic software project report writer.")

    t1 = Task(description=f"""Analyze this project: {brief}
Create scope, assumptions, actors, 8+ functional requirements with IDs, non-functional requirements,
business rules, use cases, acceptance criteria and out-of-scope items.""",
              expected_output="Structured requirements specification in Markdown.", agent=req)
    t2 = Task(description=f"""Design the architecture for: {brief}
Use previous requirements. Include modules, data flow, database tables/fields/keys, relationships,
interfaces, security, error handling and a Mermaid diagram.""",
              expected_output="Architecture and database design in Markdown.", agent=arch, context=[t1])
    t3 = Task(description=f"""Generate a small prototype for: {brief}
Follow prior requirements and architecture. Give a file tree and complete source code in separate
fenced code blocks. Include validation and error handling. Do not claim it was executed.
For a library system include add/view books, register students, issue and return books.""",
              expected_output="Prototype file tree and source code.", agent=dev, context=[t1, t2])
    t4 = Task(description=f"""Create tests for: {brief}
Use previous outputs. Include test strategy, test cases with IDs, steps, expected results, edge/security
cases and pytest/equivalent code. Clearly distinguish proposed tests from executed tests.""",
              expected_output="Test plan and automated test examples.", agent=qa, context=[t1, t2, t3])
    t5 = Task(description=f"""Document this project: {brief}
Use all prior outputs. Include overview, prerequisites, installation, configuration, database design,
modules, run instructions, user guide, testing, troubleshooting and limitations. Do not claim
features are implemented unless shown in the prototype.""",
              expected_output="Complete README-style documentation.", agent=docs, context=[t1, t2, t3, t4])
    t6 = Task(description=f"""Write an academic report for: {brief}
Use prior outputs. Include abstract, introduction, problem statement, objectives, existing/proposed
system, scope, requirements, architecture, implementation, testing, results, limitations, conclusion
and future scope. Use placeholders for student/college details. Never invent test results or metrics.""",
              expected_output="Complete academic project report in Markdown.", agent=report,
              context=[t1, t2, t3, t4, t5])

    crew = Crew(agents=[req, arch, dev, qa, docs, report],
                tasks=[t1, t2, t3, t4, t5, t6], process=Process.sequential, verbose=True)
    crew.kickoff()
    tasks = {"requirements": t1, "architecture": t2, "code": t3,
             "testing": t4, "documentation": t5, "report": t6}
    names = {"requirements": "01_requirements.md", "architecture": "02_architecture.md",
             "code": "03_prototype_code.md", "testing": "04_testing.md",
             "documentation": "05_documentation.md", "report": "06_project_report.md"}
    result = {}
    for key, task in tasks.items():
        raw = task.output.raw if task.output else "No output returned."
        result[key] = str(raw)
        (output_dir / names[key]).write_text(result[key], encoding="utf-8")
    return result
