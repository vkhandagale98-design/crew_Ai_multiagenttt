
# ============================================================
# CREWAI MULTI-AGENT SYSTEM FOR PROJECT DEVELOPMENT
# ============================================================

# IMPORTANT: Groq cache_breakpoint compatibility workaround
# Must run before importing CrewAI agents.

try:
    import crewai.llms.cache as crew_cache
    crew_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import zipfile
import streamlit as st

from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv(
    "GROQ_API_MODEL",
    "openai/gpt-oss-120b"
).strip()

st.set_page_config(
    page_title="CrewAI Multi-Agent Project Developer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #2446a8;
    font-size: 35px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.stButton > button {
    background-color: #2446a8;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 CrewAI Multi-Agent System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Software Project Development Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Generate project requirements, software architecture, "
    "prototype code, test cases, documentation and reports "
    "using six specialized AI agents."
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def check_groq_configuration():

    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is missing in your .env file."

    if not GROQ_API_KEY.startswith("gsk_"):
        return False, "Invalid Groq API key format."

    if not MODEL_NAME:
        return False, "GROQ_API_MODEL is missing."

    return True, "Groq API key detected successfully!"


def get_llm():

    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing.")

    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


# ============================================================
# CREATE SIX AGENTS
# ============================================================

def create_agents(llm):

    agents = {}

    agents["requirements"] = Agent(
        role="Software Requirements Analyst",
        goal=(
            "Analyze the project and prepare complete "
            "functional and non-functional requirements."
        ),
        backstory=(
            "You are an experienced business analyst "
            "specializing in software requirement analysis "
            "and SRS documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["architecture"] = Agent(
        role="Software Architect",
        goal=(
            "Design scalable software architecture, "
            "database structure and application workflow."
        ),
        backstory=(
            "You are an expert software architect experienced "
            "in databases, APIs, system design and security."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["code"] = Agent(
        role="Senior Software Developer",
        goal=(
            "Generate clean, structured and maintainable "
            "software prototype source code."
        ),
        backstory=(
            "You are an experienced software developer "
            "specializing in Python and application development."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["testing"] = Agent(
        role="Software Testing Engineer",
        goal=(
            "Create comprehensive test cases and "
            "automated testing code."
        ),
        backstory=(
            "You are an expert QA engineer specializing "
            "in unit testing and integration testing."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["documentation"] = Agent(
        role="Technical Documentation Specialist",
        goal=(
            "Prepare complete technical documentation "
            "and installation instructions."
        ),
        backstory=(
            "You are a professional technical writer "
            "experienced in README files and user guides."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["report"] = Agent(
        role="Project Report Specialist",
        goal=(
            "Prepare a professional final academic "
            "and technical project report."
        ),
        backstory=(
            "You are an academic project report specialist "
            "experienced in software project documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return agents


# ============================================================
# CREATE SIX TASKS
# ============================================================

def create_tasks(agents, data):

    title = data["title"]
    description = data["description"]
    language = data["language"]
    database = data["database"]
    features = data["features"]

    # TASK 1: REQUIREMENTS

    requirements_task = Task(
        description=f"""
        Analyze this software project.

        Title: {title}
        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Prepare a complete Software Requirements Specification.

        Include:
        1. Project overview
        2. Problem statement
        3. Objectives
        4. Functional requirements
        5. Non-functional requirements
        6. User roles
        7. System constraints
        8. Hardware and software requirements
        9. Expected outcomes
        """,
        expected_output=(
            "Detailed Software Requirements Specification "
            "in Markdown."
        ),
        agent=agents["requirements"]
    )

    # TASK 2: ARCHITECTURE

    architecture_task = Task(
        description=f"""
        Design the software architecture for {title}.

        Programming language: {language}
        Database: {database}

        Use the requirements task as input.

        Include:
        1. System architecture
        2. Architecture components
        3. Database tables and fields
        4. Application workflow
        5. API design where applicable
        6. Project folder structure
        7. Technology stack
        8. Security considerations
        9. Deployment approach
        """,
        expected_output=(
            "Complete software architecture document "
            "with database design and workflow."
        ),
        agent=agents["architecture"],
        context=[requirements_task]
    )

    # TASK 3: CODE GENERATION

    code_task = Task(
        description=f"""
        Develop a software prototype for {title}.

        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Follow the requirements and architecture.

        Include:
        1. Complete source code
        2. Main application file
        3. Database integration
        4. Input validation
        5. Error handling
        6. Project folder structure
        7. Requirements.txt
        8. Comments

        Show each filename and its corresponding code
        in separate fenced code blocks.

        Do not claim the code was executed or tested.
        """,
        expected_output=(
            "Source code with filenames, folder structure "
            "and implementation details."
        ),
        agent=agents["code"],
        context=[requirements_task, architecture_task]
    )

    # TASK 4: TESTING

    testing_task = Task(
        description=f"""
        Prepare software testing for {title}.

        Analyze the requirements, architecture and code.

        Include:
        1. Unit testing
        2. Integration testing
        3. Functional testing
        4. Input validation testing
        5. Error handling testing
        6. Test cases with expected results
        7. Python pytest code

        Clearly distinguish proposed tests from
        tests that have actually been executed.
        """,
        expected_output=(
            "Complete testing strategy, test cases "
            "and pytest code."
        ),
        agent=agents["testing"],
        context=[
            requirements_task,
            architecture_task,
            code_task
        ]
    )

    # TASK 5: DOCUMENTATION

    documentation_task = Task(
        description=f"""
        Prepare complete README documentation for {title}.

        Include:
        1. Introduction
        2. Project objectives
        3. Features
        4. Technology stack
        5. Installation steps
        6. Dependency installation
        7. Environment setup
        8. How to run the application
        9. Folder structure
        10. Usage instructions
        11. Troubleshooting
        12. Future improvements

        Make it beginner-friendly.
        """,
        expected_output="Complete README.md documentation.",
        agent=agents["documentation"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task
        ]
    )

    # TASK 6: FINAL REPORT

    report_task = Task(
        description=f"""
        Prepare a professional project report.

        Project: {title}
        Description: {description}

        Use the results of all previous tasks.

        Include:
        1. Abstract
        2. Introduction
        3. Problem statement
        4. Objectives
        5. Background study
        6. Requirements analysis
        7. System architecture
        8. Implementation
        9. Testing methodology
        10. Expected results
        11. Advantages
        12. Limitations
        13. Future scope
        14. Conclusion
        15. References

        Do not invent actual test results.
        """,
        expected_output="Complete final project report.",
        agent=agents["report"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task,
            documentation_task
        ]
    )

    return [
        requirements_task,
        architecture_task,
        code_task,
        testing_task,
        documentation_task,
        report_task
    ]


# ============================================================
# RUN THE CREW
# ============================================================

def run_project_crew(data):

    llm = get_llm()

    agents = create_agents(llm)

    tasks = create_tasks(agents, data)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    outputs = {}

    for name, task in zip(names, tasks):

        if task.output is not None:
            outputs[name] = (
                task.output.raw
                if task.output.raw is not None
                else str(task.output)
            )
        else:
            outputs[name] = "No output generated."

    return result, outputs


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(title, outputs):

    safe_title = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        title
    ).strip("_") or "project"

    folder = OUTPUT_DIR / safe_title
    folder.mkdir(parents=True, exist_ok=True)

    for name, content in outputs.items():

        (folder / f"{name}.md").write_text(
            str(content),
            encoding="utf-8"
        )

    (folder / "project_outputs.json").write_text(
        json.dumps(
            outputs,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return folder


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.subheader("Groq API Configuration")

    api_ok, api_message = check_groq_configuration()

    if api_ok:
        st.success(api_message)
    else:
        st.error(api_message)

    st.caption(f"Model: {MODEL_NAME}")

    st.divider()

    st.subheader("Six AI Agents")

    for agent_name in [
        "📋 Requirements Analysis",
        "🏗️ Software Architecture",
        "💻 Code Generation",
        "🧪 Software Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]:
        st.write(agent_name)


# ============================================================
# PROJECT INPUT FORM
# ============================================================

st.header("📝 Project Information")

with st.form("project_form"):

    project_title = st.text_input(
        "Project Title",
        placeholder="Example: AI Career Guidance System"
    )

    description = st.text_area(
        "Project Description",
        height=120,
        placeholder="Describe your project..."
    )

    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Programming Language",
            ["Python", "Java", "JavaScript", "C++", "PHP", "Other"]
        )

    with col2:
        database = st.selectbox(
            "Database",
            [
                "SQLite",
                "MySQL",
                "PostgreSQL",
                "MongoDB",
                "Firebase",
                "None"
            ]
        )

    features = st.text_area(
        "Required Features",
        height=130,
        placeholder=(
            "1. User registration\n"
            "2. Login system\n"
            "3. Dashboard\n"
            "4. Reports"
        )
    )

    submitted = st.form_submit_button(
        "🚀 Generate Complete Project",
        use_container_width=True
    )


# ============================================================
# GENERATE PROJECT
# ============================================================

if submitted:

    if not project_title.strip():
        st.error("Please enter a project title.")

    elif not description.strip():
        st.error("Please enter a project description.")

    elif not features.strip():
        st.error("Please enter project features.")

    elif not api_ok:
        st.error(api_message)

    else:

        data = {
            "title": project_title.strip(),
            "description": description.strip(),
            "language": language,
            "database": database,
            "features": features.strip()
        }

        try:

            with st.spinner(
                "🤖 Six AI agents are working. Please wait..."
            ):

                result, outputs = run_project_crew(data)

                folder = save_outputs(
                    project_title,
                    outputs
                )

            st.session_state["project_outputs"] = outputs
            st.session_state["project_folder"] = str(folder)
            st.session_state["project_title"] = project_title

            st.success("All six agent tasks completed!")
            st.balloons()

        except Exception as exc:

            st.error(
                f"Generation failed: {type(exc).__name__}: {exc}"
            )

            with st.expander("Technical error details"):
                st.exception(exc)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "project_outputs" in st.session_state:

    outputs = st.session_state["project_outputs"]

    st.divider()

    st.header("📂 Generated Project Results")

    st.write(
        f"Project: **{st.session_state['project_title']}**"
    )

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    labels = [
        "📋 Requirements",
        "🏗️ Architecture",
        "💻 Source Code",
        "🧪 Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]

    tabs = st.tabs(labels)

    for tab, name in zip(tabs, names):

        with tab:

            content = outputs.get(
                name,
                "No output available."
            )

            st.markdown(content)

            st.download_button(
                label=f"Download {name.title()}",
                data=content,
                file_name=f"{name}.md",
                mime="text/markdown",
                key=f"download_{name}"
            )

    # ========================================================
    # DOWNLOAD COMPLETE ZIP
    # ========================================================

    st.divider()

    st.header("📦 Download Complete Project")

    folder = Path(st.session_state["project_folder"])

    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for file_path in folder.rglob("*"):

            if file_path.is_file():

                zip_file.write(
                    file_path,
                    arcname=file_path.relative_to(folder)
                )

    st.download_button(
        label="📥 Download Complete Project ZIP",
        data=buffer.getvalue(),
        file_name="generated_project.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.info(
        "The ZIP contains generated Markdown documents and "
        "a JSON summary. Source code is provided for review "
        "and is not automatically executed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <b>CrewAI Multi-Agent System for Project Development</b>
        <br>
        Python | Streamlit | CrewAI | Groq
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CREWAI MULTI-AGENT SYSTEM FOR PROJECT DEVELOPMENT
# ============================================================

# IMPORTANT: Groq cache_breakpoint compatibility workaround
# Must run before importing CrewAI agents.

try:
    import crewai.llms.cache as crew_cache
    crew_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import zipfile
import streamlit as st

from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv(
    "GROQ_API_MODEL",
    "openai/gpt-oss-120b"
).strip()

st.set_page_config(
    page_title="CrewAI Multi-Agent Project Developer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #2446a8;
    font-size: 35px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.stButton > button {
    background-color: #2446a8;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 CrewAI Multi-Agent System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Software Project Development Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Generate project requirements, software architecture, "
    "prototype code, test cases, documentation and reports "
    "using six specialized AI agents."
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def check_groq_configuration():

    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is missing in your .env file."

    if not GROQ_API_KEY.startswith("gsk_"):
        return False, "Invalid Groq API key format."

    if not MODEL_NAME:
        return False, "GROQ_API_MODEL is missing."

    return True, "Groq API key detected successfully!"


def get_llm():

    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing.")

    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


# ============================================================
# CREATE SIX AGENTS
# ============================================================

def create_agents(llm):

    agents = {}

    agents["requirements"] = Agent(
        role="Software Requirements Analyst",
        goal=(
            "Analyze the project and prepare complete "
            "functional and non-functional requirements."
        ),
        backstory=(
            "You are an experienced business analyst "
            "specializing in software requirement analysis "
            "and SRS documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["architecture"] = Agent(
        role="Software Architect",
        goal=(
            "Design scalable software architecture, "
            "database structure and application workflow."
        ),
        backstory=(
            "You are an expert software architect experienced "
            "in databases, APIs, system design and security."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["code"] = Agent(
        role="Senior Software Developer",
        goal=(
            "Generate clean, structured and maintainable "
            "software prototype source code."
        ),
        backstory=(
            "You are an experienced software developer "
            "specializing in Python and application development."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["testing"] = Agent(
        role="Software Testing Engineer",
        goal=(
            "Create comprehensive test cases and "
            "automated testing code."
        ),
        backstory=(
            "You are an expert QA engineer specializing "
            "in unit testing and integration testing."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["documentation"] = Agent(
        role="Technical Documentation Specialist",
        goal=(
            "Prepare complete technical documentation "
            "and installation instructions."
        ),
        backstory=(
            "You are a professional technical writer "
            "experienced in README files and user guides."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["report"] = Agent(
        role="Project Report Specialist",
        goal=(
            "Prepare a professional final academic "
            "and technical project report."
        ),
        backstory=(
            "You are an academic project report specialist "
            "experienced in software project documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return agents


# ============================================================
# CREATE SIX TASKS
# ============================================================

def create_tasks(agents, data):

    title = data["title"]
    description = data["description"]
    language = data["language"]
    database = data["database"]
    features = data["features"]

    # TASK 1: REQUIREMENTS

    requirements_task = Task(
        description=f"""
        Analyze this software project.

        Title: {title}
        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Prepare a complete Software Requirements Specification.

        Include:
        1. Project overview
        2. Problem statement
        3. Objectives
        4. Functional requirements
        5. Non-functional requirements
        6. User roles
        7. System constraints
        8. Hardware and software requirements
        9. Expected outcomes
        """,
        expected_output=(
            "Detailed Software Requirements Specification "
            "in Markdown."
        ),
        agent=agents["requirements"]
    )

    # TASK 2: ARCHITECTURE

    architecture_task = Task(
        description=f"""
        Design the software architecture for {title}.

        Programming language: {language}
        Database: {database}

        Use the requirements task as input.

        Include:
        1. System architecture
        2. Architecture components
        3. Database tables and fields
        4. Application workflow
        5. API design where applicable
        6. Project folder structure
        7. Technology stack
        8. Security considerations
        9. Deployment approach
        """,
        expected_output=(
            "Complete software architecture document "
            "with database design and workflow."
        ),
        agent=agents["architecture"],
        context=[requirements_task]
    )

    # TASK 3: CODE GENERATION

    code_task = Task(
        description=f"""
        Develop a software prototype for {title}.

        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Follow the requirements and architecture.

        Include:
        1. Complete source code
        2. Main application file
        3. Database integration
        4. Input validation
        5. Error handling
        6. Project folder structure
        7. Requirements.txt
        8. Comments

        Show each filename and its corresponding code
        in separate fenced code blocks.

        Do not claim the code was executed or tested.
        """,
        expected_output=(
            "Source code with filenames, folder structure "
            "and implementation details."
        ),
        agent=agents["code"],
        context=[requirements_task, architecture_task]
    )

    # TASK 4: TESTING

    testing_task = Task(
        description=f"""
        Prepare software testing for {title}.

        Analyze the requirements, architecture and code.

        Include:
        1. Unit testing
        2. Integration testing
        3. Functional testing
        4. Input validation testing
        5. Error handling testing
        6. Test cases with expected results
        7. Python pytest code

        Clearly distinguish proposed tests from
        tests that have actually been executed.
        """,
        expected_output=(
            "Complete testing strategy, test cases "
            "and pytest code."
        ),
        agent=agents["testing"],
        context=[
            requirements_task,
            architecture_task,
            code_task
        ]
    )

    # TASK 5: DOCUMENTATION

    documentation_task = Task(
        description=f"""
        Prepare complete README documentation for {title}.

        Include:
        1. Introduction
        2. Project objectives
        3. Features
        4. Technology stack
        5. Installation steps
        6. Dependency installation
        7. Environment setup
        8. How to run the application
        9. Folder structure
        10. Usage instructions
        11. Troubleshooting
        12. Future improvements

        Make it beginner-friendly.
        """,
        expected_output="Complete README.md documentation.",
        agent=agents["documentation"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task
        ]
    )

    # TASK 6: FINAL REPORT

    report_task = Task(
        description=f"""
        Prepare a professional project report.

        Project: {title}
        Description: {description}

        Use the results of all previous tasks.

        Include:
        1. Abstract
        2. Introduction
        3. Problem statement
        4. Objectives
        5. Background study
        6. Requirements analysis
        7. System architecture
        8. Implementation
        9. Testing methodology
        10. Expected results
        11. Advantages
        12. Limitations
        13. Future scope
        14. Conclusion
        15. References

        Do not invent actual test results.
        """,
        expected_output="Complete final project report.",
        agent=agents["report"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task,
            documentation_task
        ]
    )

    return [
        requirements_task,
        architecture_task,
        code_task,
        testing_task,
        documentation_task,
        report_task
    ]


# ============================================================
# RUN THE CREW
# ============================================================

def run_project_crew(data):

    llm = get_llm()

    agents = create_agents(llm)

    tasks = create_tasks(agents, data)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    outputs = {}

    for name, task in zip(names, tasks):

        if task.output is not None:
            outputs[name] = (
                task.output.raw
                if task.output.raw is not None
                else str(task.output)
            )
        else:
            outputs[name] = "No output generated."

    return result, outputs


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(title, outputs):

    safe_title = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        title
    ).strip("_") or "project"

    folder = OUTPUT_DIR / safe_title
    folder.mkdir(parents=True, exist_ok=True)

    for name, content in outputs.items():

        (folder / f"{name}.md").write_text(
            str(content),
            encoding="utf-8"
        )

    (folder / "project_outputs.json").write_text(
        json.dumps(
            outputs,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return folder


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.subheader("Groq API Configuration")

    api_ok, api_message = check_groq_configuration()

    if api_ok:
        st.success(api_message)
    else:
        st.error(api_message)

    st.caption(f"Model: {MODEL_NAME}")

    st.divider()

    st.subheader("Six AI Agents")

    for agent_name in [
        "📋 Requirements Analysis",
        "🏗️ Software Architecture",
        "💻 Code Generation",
        "🧪 Software Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]:
        st.write(agent_name)


# ============================================================
# PROJECT INPUT FORM
# ============================================================

st.header("📝 Project Information")

with st.form("project_form"):

    project_title = st.text_input(
        "Project Title",
        placeholder="Example: AI Career Guidance System"
    )

    description = st.text_area(
        "Project Description",
        height=120,
        placeholder="Describe your project..."
    )

    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Programming Language",
            ["Python", "Java", "JavaScript", "C++", "PHP", "Other"]
        )

    with col2:
        database = st.selectbox(
            "Database",
            [
                "SQLite",
                "MySQL",
                "PostgreSQL",
                "MongoDB",
                "Firebase",
                "None"
            ]
        )

    features = st.text_area(
        "Required Features",
        height=130,
        placeholder=(
            "1. User registration\n"
            "2. Login system\n"
            "3. Dashboard\n"
            "4. Reports"
        )
    )

    submitted = st.form_submit_button(
        "🚀 Generate Complete Project",
        use_container_width=True
    )


# ============================================================
# GENERATE PROJECT
# ============================================================

if submitted:

    if not project_title.strip():
        st.error("Please enter a project title.")

    elif not description.strip():
        st.error("Please enter a project description.")

    elif not features.strip():
        st.error("Please enter project features.")

    elif not api_ok:
        st.error(api_message)

    else:

        data = {
            "title": project_title.strip(),
            "description": description.strip(),
            "language": language,
            "database": database,
            "features": features.strip()
        }

        try:

            with st.spinner(
                "🤖 Six AI agents are working. Please wait..."
            ):

                result, outputs = run_project_crew(data)

                folder = save_outputs(
                    project_title,
                    outputs
                )

            st.session_state["project_outputs"] = outputs
            st.session_state["project_folder"] = str(folder)
            st.session_state["project_title"] = project_title

            st.success("All six agent tasks completed!")
            st.balloons()

        except Exception as exc:

            st.error(
                f"Generation failed: {type(exc).__name__}: {exc}"
            )

            with st.expander("Technical error details"):
                st.exception(exc)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "project_outputs" in st.session_state:

    outputs = st.session_state["project_outputs"]

    st.divider()

    st.header("📂 Generated Project Results")

    st.write(
        f"Project: **{st.session_state['project_title']}**"
    )

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    labels = [
        "📋 Requirements",
        "🏗️ Architecture",
        "💻 Source Code",
        "🧪 Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]

    tabs = st.tabs(labels)

    for tab, name in zip(tabs, names):

        with tab:

            content = outputs.get(
                name,
                "No output available."
            )

            st.markdown(content)

            st.download_button(
                label=f"Download {name.title()}",
                data=content,
                file_name=f"{name}.md",
                mime="text/markdown",
                key=f"download_{name}"
            )

    # ========================================================
    # DOWNLOAD COMPLETE ZIP
    # ========================================================

    st.divider()

    st.header("📦 Download Complete Project")

    folder = Path(st.session_state["project_folder"])

    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for file_path in folder.rglob("*"):

            if file_path.is_file():

                zip_file.write(
                    file_path,
                    arcname=file_path.relative_to(folder)
                )

    st.download_button(
        label="📥 Download Complete Project ZIP",
        data=buffer.getvalue(),
        file_name="generated_project.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.info(
        "The ZIP contains generated Markdown documents and "
        "a JSON summary. Source code is provided for review "
        "and is not automatically executed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <b>CrewAI Multi-Agent System for Project Development</b>
        <br>
        Python | Streamlit | CrewAI | Groq
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CREWAI MULTI-AGENT SYSTEM FOR PROJECT DEVELOPMENT
# ============================================================

# IMPORTANT: Groq cache_breakpoint compatibility workaround
# Must run before importing CrewAI agents.

try:
    import crewai.llms.cache as crew_cache
    crew_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import zipfile
import streamlit as st

from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv(
    "GROQ_API_MODEL",
    "openai/gpt-oss-120b"
).strip()

st.set_page_config(
    page_title="CrewAI Multi-Agent Project Developer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #2446a8;
    font-size: 35px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.stButton > button {
    background-color: #2446a8;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 CrewAI Multi-Agent System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Software Project Development Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Generate project requirements, software architecture, "
    "prototype code, test cases, documentation and reports "
    "using six specialized AI agents."
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def check_groq_configuration():

    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is missing in your .env file."

    if not GROQ_API_KEY.startswith("gsk_"):
        return False, "Invalid Groq API key format."

    if not MODEL_NAME:
        return False, "GROQ_API_MODEL is missing."

    return True, "Groq API key detected successfully!"


def get_llm():

    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing.")

    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


# ============================================================
# CREATE SIX AGENTS
# ============================================================

def create_agents(llm):

    agents = {}

    agents["requirements"] = Agent(
        role="Software Requirements Analyst",
        goal=(
            "Analyze the project and prepare complete "
            "functional and non-functional requirements."
        ),
        backstory=(
            "You are an experienced business analyst "
            "specializing in software requirement analysis "
            "and SRS documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["architecture"] = Agent(
        role="Software Architect",
        goal=(
            "Design scalable software architecture, "
            "database structure and application workflow."
        ),
        backstory=(
            "You are an expert software architect experienced "
            "in databases, APIs, system design and security."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["code"] = Agent(
        role="Senior Software Developer",
        goal=(
            "Generate clean, structured and maintainable "
            "software prototype source code."
        ),
        backstory=(
            "You are an experienced software developer "
            "specializing in Python and application development."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["testing"] = Agent(
        role="Software Testing Engineer",
        goal=(
            "Create comprehensive test cases and "
            "automated testing code."
        ),
        backstory=(
            "You are an expert QA engineer specializing "
            "in unit testing and integration testing."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["documentation"] = Agent(
        role="Technical Documentation Specialist",
        goal=(
            "Prepare complete technical documentation "
            "and installation instructions."
        ),
        backstory=(
            "You are a professional technical writer "
            "experienced in README files and user guides."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    agents["report"] = Agent(
        role="Project Report Specialist",
        goal=(
            "Prepare a professional final academic "
            "and technical project report."
        ),
        backstory=(
            "You are an academic project report specialist "
            "experienced in software project documentation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return agents


# ============================================================
# CREATE SIX TASKS
# ============================================================

def create_tasks(agents, data):

    title = data["title"]
    description = data["description"]
    language = data["language"]
    database = data["database"]
    features = data["features"]

    # TASK 1: REQUIREMENTS

    requirements_task = Task(
        description=f"""
        Analyze this software project.

        Title: {title}
        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Prepare a complete Software Requirements Specification.

        Include:
        1. Project overview
        2. Problem statement
        3. Objectives
        4. Functional requirements
        5. Non-functional requirements
        6. User roles
        7. System constraints
        8. Hardware and software requirements
        9. Expected outcomes
        """,
        expected_output=(
            "Detailed Software Requirements Specification "
            "in Markdown."
        ),
        agent=agents["requirements"]
    )

    # TASK 2: ARCHITECTURE

    architecture_task = Task(
        description=f"""
        Design the software architecture for {title}.

        Programming language: {language}
        Database: {database}

        Use the requirements task as input.

        Include:
        1. System architecture
        2. Architecture components
        3. Database tables and fields
        4. Application workflow
        5. API design where applicable
        6. Project folder structure
        7. Technology stack
        8. Security considerations
        9. Deployment approach
        """,
        expected_output=(
            "Complete software architecture document "
            "with database design and workflow."
        ),
        agent=agents["architecture"],
        context=[requirements_task]
    )

    # TASK 3: CODE GENERATION

    code_task = Task(
        description=f"""
        Develop a software prototype for {title}.

        Description: {description}
        Language: {language}
        Database: {database}
        Features: {features}

        Follow the requirements and architecture.

        Include:
        1. Complete source code
        2. Main application file
        3. Database integration
        4. Input validation
        5. Error handling
        6. Project folder structure
        7. Requirements.txt
        8. Comments

        Show each filename and its corresponding code
        in separate fenced code blocks.

        Do not claim the code was executed or tested.
        """,
        expected_output=(
            "Source code with filenames, folder structure "
            "and implementation details."
        ),
        agent=agents["code"],
        context=[requirements_task, architecture_task]
    )

    # TASK 4: TESTING

    testing_task = Task(
        description=f"""
        Prepare software testing for {title}.

        Analyze the requirements, architecture and code.

        Include:
        1. Unit testing
        2. Integration testing
        3. Functional testing
        4. Input validation testing
        5. Error handling testing
        6. Test cases with expected results
        7. Python pytest code

        Clearly distinguish proposed tests from
        tests that have actually been executed.
        """,
        expected_output=(
            "Complete testing strategy, test cases "
            "and pytest code."
        ),
        agent=agents["testing"],
        context=[
            requirements_task,
            architecture_task,
            code_task
        ]
    )

    # TASK 5: DOCUMENTATION

    documentation_task = Task(
        description=f"""
        Prepare complete README documentation for {title}.

        Include:
        1. Introduction
        2. Project objectives
        3. Features
        4. Technology stack
        5. Installation steps
        6. Dependency installation
        7. Environment setup
        8. How to run the application
        9. Folder structure
        10. Usage instructions
        11. Troubleshooting
        12. Future improvements

        Make it beginner-friendly.
        """,
        expected_output="Complete README.md documentation.",
        agent=agents["documentation"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task
        ]
    )

    # TASK 6: FINAL REPORT

    report_task = Task(
        description=f"""
        Prepare a professional project report.

        Project: {title}
        Description: {description}

        Use the results of all previous tasks.

        Include:
        1. Abstract
        2. Introduction
        3. Problem statement
        4. Objectives
        5. Background study
        6. Requirements analysis
        7. System architecture
        8. Implementation
        9. Testing methodology
        10. Expected results
        11. Advantages
        12. Limitations
        13. Future scope
        14. Conclusion
        15. References

        Do not invent actual test results.
        """,
        expected_output="Complete final project report.",
        agent=agents["report"],
        context=[
            requirements_task,
            architecture_task,
            code_task,
            testing_task,
            documentation_task
        ]
    )

    return [
        requirements_task,
        architecture_task,
        code_task,
        testing_task,
        documentation_task,
        report_task
    ]


# ============================================================
# RUN THE CREW
# ============================================================

def run_project_crew(data):

    llm = get_llm()

    agents = create_agents(llm)

    tasks = create_tasks(agents, data)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    outputs = {}

    for name, task in zip(names, tasks):

        if task.output is not None:
            outputs[name] = (
                task.output.raw
                if task.output.raw is not None
                else str(task.output)
            )
        else:
            outputs[name] = "No output generated."

    return result, outputs


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(title, outputs):

    safe_title = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        title
    ).strip("_") or "project"

    folder = OUTPUT_DIR / safe_title
    folder.mkdir(parents=True, exist_ok=True)

    for name, content in outputs.items():

        (folder / f"{name}.md").write_text(
            str(content),
            encoding="utf-8"
        )

    (folder / "project_outputs.json").write_text(
        json.dumps(
            outputs,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return folder


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.subheader("Groq API Configuration")

    api_ok, api_message = check_groq_configuration()

    if api_ok:
        st.success(api_message)
    else:
        st.error(api_message)

    st.caption(f"Model: {MODEL_NAME}")

    st.divider()

    st.subheader("Six AI Agents")

    for agent_name in [
        "📋 Requirements Analysis",
        "🏗️ Software Architecture",
        "💻 Code Generation",
        "🧪 Software Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]:
        st.write(agent_name)


# ============================================================
# PROJECT INPUT FORM
# ============================================================

st.header("📝 Project Information")

with st.form("project_form"):

    project_title = st.text_input(
        "Project Title",
        placeholder="Example: AI Career Guidance System"
    )

    description = st.text_area(
        "Project Description",
        height=120,
        placeholder="Describe your project..."
    )

    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Programming Language",
            ["Python", "Java", "JavaScript", "C++", "PHP", "Other"]
        )

    with col2:
        database = st.selectbox(
            "Database",
            [
                "SQLite",
                "MySQL",
                "PostgreSQL",
                "MongoDB",
                "Firebase",
                "None"
            ]
        )

    features = st.text_area(
        "Required Features",
        height=130,
        placeholder=(
            "1. User registration\n"
            "2. Login system\n"
            "3. Dashboard\n"
            "4. Reports"
        )
    )

    submitted = st.form_submit_button(
        "🚀 Generate Complete Project",
        use_container_width=True
    )


# ============================================================
# GENERATE PROJECT
# ============================================================

if submitted:

    if not project_title.strip():
        st.error("Please enter a project title.")

    elif not description.strip():
        st.error("Please enter a project description.")

    elif not features.strip():
        st.error("Please enter project features.")

    elif not api_ok:
        st.error(api_message)

    else:

        data = {
            "title": project_title.strip(),
            "description": description.strip(),
            "language": language,
            "database": database,
            "features": features.strip()
        }

        try:

            with st.spinner(
                "🤖 Six AI agents are working. Please wait..."
            ):

                result, outputs = run_project_crew(data)

                folder = save_outputs(
                    project_title,
                    outputs
                )

            st.session_state["project_outputs"] = outputs
            st.session_state["project_folder"] = str(folder)
            st.session_state["project_title"] = project_title

            st.success("All six agent tasks completed!")
            st.balloons()

        except Exception as exc:

            st.error(
                f"Generation failed: {type(exc).__name__}: {exc}"
            )

            with st.expander("Technical error details"):
                st.exception(exc)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "project_outputs" in st.session_state:

    outputs = st.session_state["project_outputs"]

    st.divider()

    st.header("📂 Generated Project Results")

    st.write(
        f"Project: **{st.session_state['project_title']}**"
    )

    names = [
        "requirements",
        "architecture",
        "code",
        "testing",
        "documentation",
        "report"
    ]

    labels = [
        "📋 Requirements",
        "🏗️ Architecture",
        "💻 Source Code",
        "🧪 Testing",
        "📚 Documentation",
        "📊 Project Report"
    ]

    tabs = st.tabs(labels)

    for tab, name in zip(tabs, names):

        with tab:

            content = outputs.get(
                name,
                "No output available."
            )

            st.markdown(content)

            st.download_button(
                label=f"Download {name.title()}",
                data=content,
                file_name=f"{name}.md",
                mime="text/markdown",
                key=f"download_{name}"
            )

    # ========================================================
    # DOWNLOAD COMPLETE ZIP
    # ========================================================

    st.divider()

    st.header("📦 Download Complete Project")

    folder = Path(st.session_state["project_folder"])

    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for file_path in folder.rglob("*"):

            if file_path.is_file():

                zip_file.write(
                    file_path,
                    arcname=file_path.relative_to(folder)
                )

    st.download_button(
        label="📥 Download Complete Project ZIP",
        data=buffer.getvalue(),
        file_name="generated_project.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.info(
        "The ZIP contains generated Markdown documents and "
        "a JSON summary. Source code is provided for review "
        "and is not automatically executed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <b>CrewAI Multi-Agent System for Project Development</b>
        <br>
        Python | Streamlit | CrewAI | Groq
    </div>
    """,
    unsafe_allow_html=True
)
