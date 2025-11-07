# Contributing to PyMort

Thank you for considering contributing to PyMort! This document provides guidelines and instructions for contributing to the project.

## 🎯 Development Philosophy

This project enforces strict software engineering practices. All contributions must:

1. **Pass ALL quality checks** - No exceptions
2. **Include comprehensive tests** - Minimum 80% coverage
3. **Be fully typed** - MyPy runs in strict mode
4. **Follow code style** - Enforced by Black, Ruff, and isort
5. **Include documentation** - Google-style docstrings for all public APIs

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pymort.git
   cd pymort
   ```

3. **Set up development environment**:
   ```bash
   # Install uv package manager
   pip install uv
   
   # Install all development dependencies
   make install-dev
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Development Workflow

### Before You Start Coding

1. **Run initial checks** to ensure everything works:
   ```bash
   make check
   ```

2. **Understand the codebase**:
   - Read existing code and tests
   - Check type hints and documentation
   - Review the project structure

### While Coding

1. **Write tests first** (TDD approach recommended):
   ```python
   def test_new_feature():
       """Test description."""
       # Arrange
       # Act
       # Assert
   ```

2. **Add type hints** to ALL functions:
   ```python
   def calculate(value: float, factor: int = 2) -> float:
       """Calculate adjusted value."""
       return value * factor
   ```

3. **Document your code**:
   ```python
   def complex_function(
       param1: str,
       param2: Optional[int] = None
   ) -> Dict[str, Any]:
       """Brief description.
       
       Args:
           param1: Description of param1.
           param2: Description of param2. Defaults to None.
       
       Returns:
           Description of return value.
       
       Raises:
           ValueError: When invalid input provided.
       
       Examples:
           >>> complex_function("test")
           {"result": "test"}
       """
   ```

4. **Run checks frequently**:
   ```bash
   make format  # Auto-format code
   make lint    # Check for issues
   make type-check  # Verify types
   make test    # Run tests
   ```

### Before Committing

1. **Run full quality check**:
   ```bash
   make check
   ```

2. **Fix any issues**:
   ```bash
   make fix  # Auto-fix what's possible
   ```

3. **Ensure tests pass with coverage**:
   ```bash
   make test
   ```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from hypothesis import given, strategies as st

class TestFeature:
    """Test suite for Feature."""
    
    def test_basic_functionality(self):
        """Test basic feature behavior."""
        # Test implementation
        
    @pytest.mark.parametrize("input,expected", [
        ("A", 1),
        ("B", 2),
    ])
    def test_with_parameters(self, input, expected):
        """Test with different inputs."""
        # Test implementation
        
    @given(value=st.integers())
    def test_property_based(self, value):
        """Property-based test."""
        # Test implementation
```

### Coverage Requirements

- Minimum 80% code coverage
- Test both happy paths and edge cases
- Include integration tests for complex features
- Use property-based testing where appropriate

## 🎨 Code Style

### Python Style

- **Line length**: 100 characters maximum
- **Imports**: Sorted with isort
- **Formatting**: Black with default settings
- **Linting**: Ruff with extensive rules
- **Docstrings**: Google style

### Example Code

```python
"""Module docstring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pymort.base import BaseClass


class ExampleClass(BaseClass):
    """Example class with proper style.
    
    Attributes:
        name: The name of the instance.
        value: The current value.
    """
    
    def __init__(self, name: str, value: float = 0.0) -> None:
        """Initialize ExampleClass.
        
        Args:
            name: Instance name.
            value: Initial value. Defaults to 0.0.
        """
        self.name = name
        self.value = value
    
    def process(self, data: List[float]) -> np.ndarray:
        """Process data.
        
        Args:
            data: Input data to process.
            
        Returns:
            Processed data as numpy array.
            
        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        return np.array(data) * self.value
```

## 📊 Type Checking

### Strict MyPy Configuration

All code must pass MyPy in strict mode:

```python
# Good - Fully typed
def calculate_average(values: List[float]) -> float:
    """Calculate average of values."""
    if not values:
        raise ValueError("Cannot calculate average of empty list")
    return sum(values) / len(values)

# Bad - Missing type hints
def calculate_average(values):
    return sum(values) / len(values)
```

### Common Type Patterns

```python
from typing import (
    Any, Callable, Dict, List, Literal, 
    Optional, Protocol, Tuple, TypeVar, Union
)

T = TypeVar("T")

class Processor(Protocol):
    """Protocol for processors."""
    
    def process(self, data: Any) -> Any: ...

def generic_function(
    items: List[T],
    processor: Callable[[T], T],
    config: Optional[Dict[str, Any]] = None
) -> List[T]:
    """Process items with given processor."""
    config = config or {}
    return [processor(item) for item in items]
```

## 🔄 Pull Request Process

1. **Ensure all checks pass**:
   ```bash
   make ci  # Simulate full CI pipeline
   ```

2. **Update documentation** if needed

3. **Write clear commit messages**:
   ```
   feat: add new strategy implementation
   
   - Implement AlwaysCooperate strategy
   - Add comprehensive tests
   - Update documentation
   
   Closes #123
   ```

4. **Create Pull Request**:
   - Use descriptive title
   - Fill out PR template
   - Link related issues
   - Request review from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass with >80% coverage
- [ ] Type checking passes (MyPy strict mode)
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass
- [ ] No security vulnerabilities
- [ ] PR description explains changes

## 🐛 Reporting Issues

### Bug Reports

Include:
- Python version
- OS information
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/traceback

### Feature Requests

Include:
- Use case description
- Proposed implementation
- Alternative solutions considered
- Impact on existing code

## 🏗️ Project Structure

```
src/pymort/        # Main package
├── __init__.py    # Public API exports
├── cli.py         # CLI interface
├── triangle.py    # Triangle data structures
├── mack.py        # Mack chain-ladder implementation
└── bootstrap.py   # Bootstrap simulation

tests/             # Test suite
├── conftest.py    # Pytest fixtures
├── test_*.py      # Test modules
└── data/          # Test data

docs/              # Documentation
├── api/           # API documentation
└── guides/        # User guides
```

## 🛠️ Development Tools

### Essential Commands

```bash
make help          # Show all commands
make check         # Run all checks
make fix           # Auto-fix issues
make test          # Run tests
make ci            # Simulate CI
```

### Debugging

```bash
# Detailed type errors
mypy src/ --show-error-codes --pretty

# Verbose test output
pytest -vvs

# Profile performance
python -m cProfile -s cumulative script.py

# Check coverage gaps
make test && open htmlcov/index.html
```

## 📜 Code of Conduct

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards other contributors

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🙏 Thank You!

Your contributions make this project better. We appreciate your time and effort in maintaining high code quality standards!