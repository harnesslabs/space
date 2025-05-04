# Contributing to Harness

Thank you for your interest in contributing to Harness! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/harness.git
   cd harness
   ```
3. Set up the development environment:
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install development tools
   rustup component add rustfmt clippy
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b type/area/description
   # Example: git checkout -b feat/algebra/vector-spaces
   ```

2. Make your changes following the [Code Style](#code-style) guidelines

3. Run tests and checks:
   ```bash
   cargo test
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   ```

4. Commit your changes following the [Commit Message Format](#commit-message-format)

5. Push your branch and create a Pull Request

## Issue Guidelines

When creating issues, please use the provided templates and follow these guidelines:

### Title Format
```
type(area): brief description
```
Where:
- `type` is one of: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- `area` is one of: `algebra`, `space`

### Labels
Please use appropriate labels to categorize your issue:
- Area labels: `area: algebra`, `area: space`
- Priority labels: `priority: critical/high/medium/low`
- Type labels: `type: enhancement`, `type: refactor`
- Technical labels: `tech: performance`, `tech: security`, `tech: testing`

## Pull Request Guidelines

1. Use the provided PR template
2. Ensure your PR title follows the format: `type(area): description`
3. Link related issues using `closes #issue_number`
4. Keep PRs focused and small when possible
5. Include tests for new features or bug fixes
6. Update documentation as needed

## Code Style

- Follow Rust's official style guide
- Use `rustfmt` for formatting
- Run `cargo clippy` to catch common mistakes
- Document public APIs thoroughly
- Use meaningful variable and function names
- Keep functions focused and small

## Testing

- Write unit tests for all new functionality
- Include examples in documentation
- Run all tests before submitting PRs
- Consider edge cases and error conditions

## Documentation

- Document all public APIs
- Include mathematical definitions and references
- Provide usage examples
- Keep documentation up to date with code changes

## Commit Message Format

Follow this format for commit messages:
```
type(area): description

[optional body]

[optional footer]
```

Where:
- `type` is one of: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- `area` is one of: `algebra`, `space`
- Description is a brief summary of changes
- Body provides additional context if needed
- Footer references issues or PRs

## Questions?

If you have any questions, feel free to:
1. Open an issue with the `question` label
2. Join our community discussions
3. Contact the maintainers 