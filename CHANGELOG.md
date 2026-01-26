# Changelog

All notable changes to the **Parishad** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-01-26

### Fixed
- **Critical Crash**: Fixed `TypeError` in `PerceptionTool` initialization by updating the constructor to accept `llm_config`.

## [0.1.1] - 2026-01-26

### Fixed
- **Windows Dependencies**: Added `textual` and `psutil` to package requirements.
- **Model Downloads**: Fixed 404 errors by properly handling `-GGUF` suffixes in repository names.

## [0.1.0] - Initial Release

### Added
- **Core Architecture**: Structured Council of LLMs (Darbari, Majumdar, Sainik, Prerak, Raja).
- **Interactive TUI**: Real-time terminal dashboard with visual role tracking.
- **Unified CLI**: Single `parishad` command handles setup, permissions, and execution.
- **Slash Commands**: `/sabha`, `/scan`, `/roles`, `/history`, `/clear`.
- **Context Awareness**: Support for `@path/to/file` context injection in queries.
- **Vision**: `PerceptionTool` for analyzing images with local VLMs.
- **Local-First Backends**: Support for Ollama, Llama.cpp, MLX, and Transformers.
- **Documentation**: Professional README, contributing guidelines, and security policy.

### Infrastructure
- **Cost Tracking**: Configurable token budgets per query.
- **Adaptive Routing**: Auto-selects model size (Small/Mid/Big) based on task complexity.
