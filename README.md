<p align="center">
  <img src="https://raw.githubusercontent.com/harnesslabs/brand/main/cova/cova-banner.png" alt="Cova Banner" width="500">
</p>

[![Crates.io - cova](https://img.shields.io/crates/v/cova?label=cova)](https://crates.io/crates/cova)
[![docs.rs - cova](https://img.shields.io/docsrs/cova?label=docs.rs%20cova)](https://docs.rs/cova)
[![Crates.io - cova-space](https://img.shields.io/crates/v/cova-space?label=cova-space)](https://crates.io/crates/cova-space)
[![docs.rs - cova-space](https://img.shields.io/docsrs/cova-space?label=docs.rs%20cova-space)](https://docs.rs/cova-space)
[![Crates.io - cova-algebra](https://img.shields.io/crates/v/cova-algebra?label=cova-algebra)](https://crates.io/crates/cova-algebra)
[![docs.rs - cova-algebra](https://img.shields.io/docsrs/cova-algebra?label=docs.rs%20cova-algebra)](https://docs.rs/cova-algebra)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Cova

A Rust ecosystem for mathematical abstractions and computations, focusing on rigorous implementations of algebraic structures, topological spaces, and computational mathematics.

## Overview

Cova provides a collection of crates that implement various mathematical structures and algorithms with a focus on type safety, correctness, and composability. The project aims to provide foundational mathematical tools that can be used in scientific computing, computational topology, abstract algebra, and other domains requiring robust mathematical implementations.

## Examples & Demos

Cova includes interactive demos to help you get started:

### 🌐 Interactive Web Demos

- **[Vietoris-Rips Complex Demo](examples/vietoris_web/README.md)**: An interactive WebAssembly demo showcasing real-time topological data analysis. Click to place points and watch simplicial complexes emerge as you adjust the distance threshold.

## Design Philosophy

- **Type Safety**: Mathematical properties are encoded in the type system where possible
- **Correctness**: Implementations prioritize mathematical correctness over performance
- **Composability**: Structures are designed to work together seamlessly
- **Documentation**: Extensive mathematical documentation and examples

## Crates

### `cova`

The `cova` crate is a meta crate that re-exports the `cova-space` and `cova-algebra` crates.


### `cova-space`

The `cova-space` crate implements topological spaces, simplicial complexes, and graph structures, providing a foundation for computational topology and geometry. It includes:

- **Topological Spaces**: Sets, metric spaces, normed spaces, and inner product spaces
- **Simplicial Complexes**: Simplex representation, chain complexes, and homology computations
- **Graph Theory**: Flexible directed and undirected graph data structures
- **Sheaf Theory**: Advanced categorical constructions for topology
- **Filtrations**: Tools for persistent homology and topological data analysis

### `cova-algebra`

The `cova-algebra` crate provides implementations of algebraic structures with proper type constraints and mathematical rigor. It includes:

- **Modular Arithmetic**: Custom modular number types with the `modular!` macro
- **Abstract Algebra**: Groups, rings, fields, modules, and vector spaces
- **Category Theory**: Fundamental categorical constructions and morphisms
- **Tensors**: Tensor algebra and operations
- **Linear Algebra**: Vector spaces and linear transformations

## Getting Started

### Prerequisites

Cova requires Rust 1.70 or later.

### Installation

Add the desired crates to your `Cargo.toml`:

```toml
[dependencies]
cova = "*"
# or if you only need one of the crates
cova-space = "*"
cova-algebra = "*" 
```

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/harnesslabs/cova.git
   cd cova
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

### Viewing Documentation

The project provides two types of documentation:

1. **API Documentation**: View the Rust API documentation for all crates:
   ```bash
   just docs
   ```
   This will build and open the Rust API documentation in your browser.

2. **Book Documentation**: View the comprehensive book documentation:
   ```bash
   just book
   ```
   This will serve the book documentation locally and open it in your browser. The book includes detailed explanations of mathematical concepts, examples, and usage guides.

For more development commands, run `just --list`.

## Documentation

- [API Documentation - cova](https://docs.rs/cova)
- [API Documentation - cova-space](https://docs.rs/cova-space)
- [API Documentation - cova-algebra](https://docs.rs/cova-algebra)
- [Book](https://book.harnesslabs.xyz)

## Contributing

We welcome contributions! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to the project.

## License

This project is licensed under the [AGPLv3 License](LICENSE).

