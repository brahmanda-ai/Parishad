# Contributing to Parishad

Thank you for your interest in contributing to **Parishad**! We welcome contributions from the community to make this "Council of LLMs" even better.

## 🚀 Getting Started

1.  **Fork the Repository**: Click the "Fork" button on the top right of the GitHub page.
2.  **Clone your Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/parishad.git
    cd parishad
    ```
3.  **Set up Development Environment**:
    ```bash
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install development dependencies
    pip install -e ".[dev]"
    ```

## 🛠️ Development Workflow

### Code Style
We follow strict code quality standards "Claude Code" style.
- **Formatter**: `black` (line length 100)
- **Linter**: `ruff`
- **Type Checking**: `mypy`

Before submitting a PR, run:
```bash
# Format code
black .

# Linting
ruff check .

# Type checking
mypy src
```

### Running Tests
Ensure all tests pass before submitting:
```bash
pytest
```

## 📝 Pull Request Guidelines

1.  **Descriptive Title**: Use a clear title (e.g., `feat: Add new 'Sacheev' advisor role`).
2.  **Description**: Explain *what* changed and *why*.
3.  **Link Issues**: If this fixes an issue, link it (e.g., `Fixes #123`).
4.  **Tests**: Include new tests for new features.

## 🐛 Reporting Issues

- Check existing issues before creating a new one.
- Use the **Bug Report** or **Feature Request** templates.
- Include reproduction steps and environment details (OS, Python version).

Thank you for helping us build the ultimate local LLM council! 🏛️
