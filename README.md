<div align="center">

![Parishad Gradient Logo](docs/assets/logo.svg)

**A cost-aware, local-first council of heterogeneous LLMs for reliable reasoning, coding, and factual correctness.**

[![PyPI Version](https://img.shields.io/pypi/v/parishad?style=flat-square&color=orange)](https://pypi.org/project/parishad/)
[![Python Version](https://img.shields.io/pypi/pyversions/parishad?style=flat-square&color=blue)](https://pypi.org/project/parishad/)
[![License](https://img.shields.io/github/license/parishad-council/parishad?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Type Checked](https://img.shields.io/badge/mypy-checked-blue.svg?style=flat-square)](http://mypy-lang.org/)

</div>

---

## 📖 Overview

**Parishad** (Hindi: *Council*) orchestrates multiple local language models into a structured reasoning pipeline. Instead of relying on a single massive model, Parishad coordinates specialized, smaller models (2B-34B parameters) into functional roles based on traditional Indian administrative titles.

This approach achieves **higher reliability** and **lower latency** on consumer hardware (Mac Silicon, NVIDIA GPUs) by assigning tasks to the most appropriate model slot (Small/Medium/Big) tailored to your compute budget.

## ✨ Key Features

- **🏛️ Heterogeneous Model Council**: Seamlessly orchestrate varying model sizes (e.g., Phi-3 for syntax, Llama-3 for reasoning, Mixtral for judgment).
- **🖥️ Beautiful TUI Dashboard**: Real-time interactive terminal interface with visual role tracking, progress bars, and emoji indicators.
- **👁️ Vision Capabilities**: Integrated `PerceptionTool` allows the council to "see" and analyze images via local VLMs.
- **💰 Cost-Aware Execution**: Define token budgets per query; the council adapts its strategy to stay within limits.
- **🛡️ Structured Verification**: The `Prerak` (Checker) role actively challenges hallucinations using deterministic tools and cross-examination.
- **🔌 Local-First Backends**: Native support for **Ollama**, **Llama.cpp**, **MLX**, and **Transformers**.

## 🏛️ The Three Sabhas (Council Configurations)

Parishad offers three distinct council sizes to balance speed, cost, and depth:

### 1. Laghu Sabha (Fast Council)
*   **Roles**: 5 (Darbari, Majumdar, Sainik, Prerak, Raja)
*   **Use Case**: Quick queries, simple code generation, factual checks.
*   **Models**: Optimized for Small/Mid models.

### 2. Madhya Sabha (Core Council)
*   **Roles**: 8 (Adds **Sar Senapati** [Coordinator], **Sacheev** [Secretary], **Dandadhyaksha** [Magistrate])
*   **Use Case**: Balanced reasoning for standard development tasks.
*   **Models**: Uses Mid/Big models for deeper analysis.

### 3. Maha Sabha (Extended Council)
*   **Roles**: 10 (Adds **Pantapradhan** [Prime Minister], **Vidushak** [Lateral Thinker])
*   **Use Case**: Complex system design, creative problem solving, and strategic planning.
*   **Models**: Full spectrum orchestration (Small + Mid + Big).

## 🏗️ Architecture

Parishad organizes models into a directed graph of roles:

```text
┌───────────┐      ┌───────────┐      ┌───────────┐
│  DARBARI  │ ──►  │ MAJUMDAR  │ ──►  │  SAINIK   │
│ (Refiner) │      │ (Planner) │      │ (Worker)  │
└───────────┘      └───────────┘      └───────────┘
                                            │
                                            ▼
┌───────────┐      ┌───────────┐      ┌───────────┐
│   RAJA    │ ◄──  │  PRERAK   │ ◄──  │ SYSTEM &  │
│  (Judge)  │      │ (Checker) │      │   TOOLS   │
└───────────┘      └───────────┘      └───────────┘
```

- **Darbari**: Normalizes vague user queries into structured task specifications.
- **Majumdar**: Decomposes complex tasks into stepwise execution plans.
- **Sainik**: Executes steps, writing code or generating content using tools.
- **Prerak**: Validates output against facts, schema, or safety policies.
- **Raja**: Synthesizes the final answer and assigns a confidence score.

## 📦 Installation

### Standard Install
```bash
pip install parishad
```

### Optimized Install Options
Choose the backend that matches your hardware:

```bash
# Apple Silicon (M1/M2/M3) - Recommended
pip install "parishad[mlx]"

# NVIDIA GPU / Local Parsing
pip install "parishad[local,cuda]"

# With Retrieval (RAG) capabilities
pip install "parishad[retrieval]"

# For Developers (Linting, Testing)
pip install "parishad[dev]"
```

## 🚀 Quick Start

### 1. Launch & Setup
Run the main command to initialize the council. This automatically handles permissions, scans for models (Ollama/HF), and launches the TUI.

```bash
parishad
```

**Pro Tip:** Use `/sabha` to switch between council sizes (e.g., `laghu` for speed, `maha` for deep reasoning) and `/scan` to find new local models.

### 2. CLI Direct Execution
Run a specific task without entering the interactive shell.

```bash
parishad run --sabha madhyam "Analyze the 'data.csv' file and plot trends"
```

### 3. Python API
Integrate Parishad into your own applications.

```python
from parishad import Parishad

# Initialize the council
council = Parishad(config="core")

# Execute a complex query
result = council.run("Design a scalable REST API architecture for a bookstore.")

print(f"Final Answer: {result.final_answer}")
print(f"Confidence: {result.confidence_score}/10")
# Full execution trace available in result.trace
```

## 🎮 Interactive Features

Unlock the full power of the council with these TUI capabilities:

### ⚡ Slash Commands
Control the session without leaving the chat interface:
- **`/sabha [mode]`**: Switch council configuration instantly (e.g., `/sabha laghu` for speed, `/sabha maha` for power).
- **`/scan`**: Re-scan your system for newly downloaded models.
- **`/roles`**: View currently active roles, their models, and slot assignments.
- **`/history`**: View past queries and load previous traces.
- **`/clear`**: Reset the conversation context.

### 📎 Context Awareness (@ Mentions)
Parishad allows you to "chat with your files". Directly reference local files in your query:
- **`@path/to/file.py`**: Reads the file and adds it to the context.
- **`@src/`**: Scans the directory structure.

Example:
> *"Analyze @src/main.py and refactor the ErrorHandler class."*


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Install dev dependencies: `pip install -e ".[dev]"`
3.  Ensure code style compliance: `black . && ruff check .`
4.  Submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/ommo007">Om Mohite</a> & <a href="https://github.com/Ashiiish-88">Ashish Prajapati</a>. Trusted by developers for local agentic workflows.</sub>
</div>
