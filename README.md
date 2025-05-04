[![Crates.io - harness-space](https://img.shields.io/crates/v/harness-space?label=harness-space)](https://crates.io/crates/harness-space)
[![docs.rs - harness-space](https://img.shields.io/docsrs/harness-space?label=docs.rs%20harness-space)](https://docs.rs/harness-space)
[![Crates.io - harness-algebra](https://img.shields.io/crates/v/harness-algebra?label=harness-algebra)](https://crates.io/crates/harness-algebra)
[![docs.rs - harness-algebra](https://img.shields.io/docsrs/harness-algebra?label=docs.rs%20harness-algebra)](https://docs.rs/harness-algebra)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Harness

A Rust ecosystem for mathematical abstractions and computations, focusing on rigorous implementations of mathematical structures and algorithms.

## Overview

Harness provides a collection of crates that implement various mathematical structures and algorithms with a focus on type safety, correctness, and composability. The project aims to provide foundational mathematical tools that can be used in scientific computing, computer graphics, machine learning, and other domains requiring robust mathematical implementations.

## Design Philosophy

- **Type Safety**: Mathematical properties are encoded in the type system where possible
- **Correctness**: Implementations prioritize mathematical correctness over performance
- **Composability**: Structures are designed to work together seamlessly
- **Documentation**: Extensive mathematical documentation and examples

## Crates

### `space`

The `space` crate implements fundamental topological and geometric structures, providing a foundation for working with spaces and their properties in a type-safe manner.

### `algebra`

The `algebra` crate will provide implementations of algebraic structures, enabling rigorous mathematical computations with proper type constraints.

## Getting Started

### Prerequisites

Harness requires Rust 1.70 or later.

### Installation

Add the desired crates to your `Cargo.toml`:

```toml
[dependencies]
harness-space = "*"
harness-algebra = "*" 
```

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/harness.git
   cd harness
   ```

2. Install `just` (if not already installed):
   ```bash
   # macOS
   brew install just
   
   # Linux
   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
   ```

3. Run the development setup:
   ```bash
   just setup
   ```

4. Build and test:
   ```bash
   just test
   ```

For more development commands, run `just --list`.

## Documentation

- [API Documentation](https://docs.rs/harness-space)
- [API Documentation](https://docs.rs/harness-algebra)

## Contributing

We welcome contributions! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to the project.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

