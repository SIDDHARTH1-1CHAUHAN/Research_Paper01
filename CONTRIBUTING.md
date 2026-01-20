# Contributing to Multi-Agent Fact-Checking Pipeline

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing code, documentation, and research insights.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Research Contributions](#research-contributions)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you are expected to uphold this code.

Key principles:
- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Maintain professional communication

---

## How to Contribute

### Types of Contributions

1. **Bug Reports:** Found a bug? Open an issue with reproduction steps
2. **Feature Requests:** Suggest new capabilities or improvements
3. **Code Contributions:** Implement new features or fix bugs
4. **Documentation:** Improve README, tutorials, or API docs
5. **Research:** Contribute benchmark results, ablation studies, or new evaluation metrics
6. **Examples:** Add usage examples or tutorials

### Before You Start

1. Check existing [issues](https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01/issues) to avoid duplicates
2. For major changes, open an issue first to discuss your proposal
3. Fork the repository and create a feature branch
4. Follow the coding standards and testing guidelines

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Ollama (for local LLM testing)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Research_Paper01.git
cd Research_Paper01/multi-agent-fact-checker

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Download Ollama model
ollama pull llama3.2:3b

# 6. Create a feature branch
git checkout -b feature/your-feature-name
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with the following tools:

- **black:** Code formatting (line length: 100)
- **ruff:** Linting and style checks
- **mypy:** Type checking

### Code Formatting

```bash
# Format code
black src/ tests/ examples/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `InputIngestionAgent`)
- **Functions/Methods:** `snake_case` (e.g., `decompose_claim()`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_QUERIES`)
- **Private:** `_leading_underscore` (e.g., `_internal_method()`)

### Documentation Strings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Short one-line description.

    Longer description with more details about what this function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_agents.py

# Run with verbose output
pytest -v tests/
```

### Writing Tests

1. **Test files:** `tests/test_<module_name>.py`
2. **Test functions:** `test_<functionality>()`
3. **Use fixtures:** Define reusable test data
4. **Mock external APIs:** Don't make real API calls in tests

Example:

```python
import pytest
from src.agents.input_ingestion import InputIngestionAgent

@pytest.fixture
def sample_claim():
    return "The Eiffel Tower is in Paris"

def test_claim_decomposition(sample_claim):
    agent = InputIngestionAgent()
    result = agent.decompose(sample_claim)
    assert len(result.subclaims) > 0
    assert result.subclaims[0].verifiability == "verifiable"
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

---

## Documentation

### Types of Documentation

1. **Code Documentation:**
   - Docstrings for all public functions/classes
   - Inline comments for complex logic

2. **User Documentation:**
   - README.md (non-technical audience)
   - RESEARCH_PAPER.md (academic audience)

3. **Technical Documentation:**
   - ARCHITECTURE.md (system design)
   - API_REFERENCE.md (code reference)

### Documentation Standards

- Clear, concise language
- Examples for all major features
- Keep documentation in sync with code
- Use diagrams where helpful

---

## Research Contributions

### Adding New Benchmarks

1. Create loader in `src/evaluation/benchmark_loader.py`
2. Add configuration to `config/benchmark_config.yaml`
3. Document dataset characteristics
4. Provide baseline results for comparison

### Reporting Results

When contributing benchmark results:

1. **Reproducibility:**
   - Provide exact configuration used
   - Random seeds for stochastic components
   - Version numbers of all dependencies

2. **Statistical Rigor:**
   - Report mean and standard deviation
   - Multiple runs (≥3) for significance
   - Confidence intervals where appropriate

3. **Format:**
   ```
   Benchmark: FEVEROUS
   Configuration: config/agent_config.yaml (no modifications)
   Model: llama3.2:3b via Ollama
   Runs: 5
   Results:
   - F1-score: 0.681 ± 0.012
   - Precision: 0.704 ± 0.015
   - Recall: 0.659 ± 0.018
   ```

### Ablation Studies

When proposing new features, include ablation studies:

- Baseline (without feature)
- With feature enabled
- Statistical significance test
- Discussion of trade-offs

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (black, ruff, mypy pass)
- [ ] All tests pass (`pytest tests/`)
- [ ] New code includes tests
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Research contribution

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)

## Related Issues
Closes #(issue number)
```

### Review Process

1. **Automated Checks:**
   - CI/CD pipeline runs tests
   - Code style checks (black, ruff)
   - Type checking (mypy)

2. **Manual Review:**
   - At least one maintainer review required
   - Address review comments
   - Request re-review after changes

3. **Merge:**
   - Squash commits for clean history
   - Update changelog
   - Delete feature branch after merge

---

## Branch Naming

- **Features:** `feature/short-description`
- **Bug fixes:** `fix/issue-number-description`
- **Documentation:** `docs/what-changed`
- **Research:** `research/experiment-name`

Examples:
- `feature/add-gemini-support`
- `fix/123-credibility-checker-crash`
- `docs/update-installation-guide`
- `research/feverous-ablation-study`

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Test additions/updates
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add Gemini support to Evidence Seeking Agent

Implements Google Gemini 1.5 Flash for content analysis in the
evidence seeking pipeline. Falls back to Ollama if API key not provided.

Closes #45
```

---

## Issue Reporting

### Bug Reports

Include:
1. **Description:** Clear summary of the bug
2. **Steps to Reproduce:**
   ```
   1. Initialize system with config X
   2. Run claim Y
   3. Observe error Z
   ```
3. **Expected Behavior:** What should happen
4. **Actual Behavior:** What actually happens
5. **Environment:**
   - OS: Windows 10 / macOS 14 / Ubuntu 22.04
   - Python version: 3.9.7
   - Ollama version: 0.1.23
6. **Logs:** Relevant error messages or stack traces

### Feature Requests

Include:
1. **Problem Statement:** What problem does this solve?
2. **Proposed Solution:** How should it work?
3. **Alternatives Considered:** Other approaches
4. **Use Cases:** Who benefits and how?

---

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release branch: `release/v0.2.0`
4. Run full test suite
5. Tag release: `git tag v0.2.0`
6. Push to GitHub: `git push origin v0.2.0`
7. Create GitHub release with notes

---

## Getting Help

- **Questions:** Open a [GitHub Discussion](https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01/discussions)
- **Bugs:** Open an [Issue](https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01/issues)
- **Security:** Email maintainers directly (see README for contact)

---

## Recognition

Contributors will be:
- Listed in project README
- Acknowledged in research papers
- Credited in release notes

Significant contributions may lead to co-authorship on academic publications.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing robust, transparent fact-checking technology!
