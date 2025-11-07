.PHONY: help install install-dev clean test lint format type-check security docs build run pre-commit all check fix

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
UV := uv
SRC_DIR := src
TEST_DIR := tests
PKG_NAME := simclt

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Quick start:$(NC)"
	@echo "  make install-dev    # Set up development environment"
	@echo "  make check         # Run all quality checks"
	@echo "  make fix           # Auto-fix code issues"
	@echo "  make test          # Run tests with coverage"

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(UV) sync --extra dev
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	$(UV) run pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned!$(NC)"

test: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(UV) run pytest $(TEST_DIR) -v --cov=$(PKG_NAME) --cov-report=term-missing --cov-report=html --cov-fail-under=80
	@echo "$(GREEN)✓ Tests passed! Coverage report: htmlcov/index.html$(NC)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(GREEN)Running tests (fast mode)...$(NC)"
	$(UV) run pytest $(TEST_DIR) -v -n auto

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(UV) run pytest-watch -- -v

lint: ## Run linting checks
	@echo "$(GREEN)Running linters...$(NC)"
	@echo "  Ruff..."
	@$(UV) run ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format-check: ## Check code formatting
	@echo "$(GREEN)Checking code format...$(NC)"
	@$(UV) run ruff format --check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Format check complete!$(NC)"

format: ## Auto-format code
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(UV) run ruff format $(SRC_DIR) $(TEST_DIR)
	@$(UV) run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Code formatted!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checker...$(NC)"
	$(UV) run mypy $(SRC_DIR) --config-file=pyproject.toml
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	@echo "  Bandit..."
	@$(UV) run bandit -r $(SRC_DIR) -ll --skip B101
	@echo "$(GREEN)✓ Security checks complete!$(NC)"

docs: ## Check documentation
	@echo "$(GREEN)Checking documentation...$(NC)"
	@echo "  Docstring coverage..."
	@$(UV) run interrogate -vv $(SRC_DIR) --fail-under 80
	@echo "$(GREEN)✓ Documentation check complete!$(NC)"

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Build complete! Check dist/ directory$(NC)"

run: ## Run the CLI application
	@echo "$(GREEN)Running simclt CLI...$(NC)"
	$(UV) run simclt --help

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	$(UV) run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks complete!$(NC)"

pre-commit-update: ## Update pre-commit hooks
	@echo "$(GREEN)Updating pre-commit hooks...$(NC)"
	$(UV) run pre-commit autoupdate
	@echo "$(GREEN)✓ Hooks updated!$(NC)"

# Composite targets
check: lint format-check type-check test security docs ## Run all quality checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"

fix: format lint-fix ## Auto-fix all possible issues
	@echo "$(GREEN)✓ Auto-fixes applied!$(NC)"

lint-fix: ## Auto-fix linting issues
	@echo "$(GREEN)Auto-fixing linting issues...$(NC)"
	@$(UV) run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Linting issues fixed!$(NC)"

all: clean install-dev check build ## Full CI pipeline
	@echo "$(GREEN)✓ Full pipeline complete!$(NC)"

# Development workflow helpers
dev-setup: install-dev ## Initial development setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@echo "$(GREEN)Creating git hooks...$(NC)"
	@echo '#!/bin/sh\nmake pre-commit' > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "$(GREEN)✓ Development setup complete!$(NC)"

# CI simulation
ci: ## Simulate CI pipeline locally
	@echo "$(GREEN)Running CI pipeline locally...$(NC)"
	@echo "Step 1/6: Linting..."
	@make lint
	@echo "Step 2/6: Format check..."
	@make format-check
	@echo "Step 3/6: Type checking..."
	@make type-check
	@echo "Step 4/6: Security..."
	@make security
	@echo "Step 5/6: Documentation..."
	@make docs
	@echo "Step 6/6: Tests..."
	@make test
	@echo "$(GREEN)✓ CI pipeline passed!$(NC)"
