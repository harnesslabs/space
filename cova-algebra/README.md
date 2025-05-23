<p align="center">
  <img src="https://raw.githubusercontent.com/harnesslabs/brand/main/cova/cova.png" alt="Cova Logo" width="400">
</p>

# Cova Algebra

A comprehensive Rust library for abstract algebra, providing rigorous implementations of algebraic structures from basic arithmetic to advanced category theory and tensor calculus.

[![Crates.io - cova-algebra](https://img.shields.io/crates/v/cova-algebra?label=cova-algebra)](https://crates.io/crates/cova-algebra)
[![docs.rs - cova-algebra](https://img.shields.io/docsrs/cova-algebra?label=docs.rs%20cova-algebra)](https://docs.rs/cova-algebra)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Overview

Cova Algebra implements the fundamental structures of abstract algebra with a focus on mathematical correctness, type safety, and composability. The crate provides a hierarchical organization of algebraic concepts, from basic arithmetic operations to advanced constructions in algebra and category theory.

## Architecture

The library is structured around core mathematical concepts, with each module building upon more fundamental structures:

### Core Modules

#### [`arithmetic`](src/arithmetic/)
Foundation layer providing basic arithmetic operations and modular arithmetic. Includes the `modular!` macro for creating custom modular number types and fundamental arithmetic traits that serve as building blocks for higher-level structures.

#### [`groups`](src/groups.rs)
Group theory implementations covering both commutative (Abelian) and non-commutative groups. Provides the fundamental structure for understanding symmetry and transformation in algebra, with proper distinctions between additive and multiplicative group operations.

#### [`rings`](src/rings.rs)
Ring theory abstractions including rings, fields, and semirings. Establishes the algebraic foundation for structures that support both addition and multiplication, with fields providing division operations for advanced algebraic computations.

#### [`modules`](src/modules/)
Module theory over rings, including vector spaces, semimodules, and specialized constructions like tropical modules. Provides the framework for linear algebra and generalizes vector spaces to work over arbitrary rings.

### Advanced Modules

#### [`algebras`](src/algebras/)
Higher-order algebraic structures that combine vector spaces with multiplication operations. Includes Boolean algebra for logical operations and Clifford algebras for geometric applications in physics and computer graphics.

#### [`tensors`](src/tensors/)
Multi-dimensional tensor implementations with both compile-time fixed dimensions and runtime dynamic sizing. Supports tensor operations fundamental to linear algebra, differential geometry, and machine learning applications.

#### [`category`](src/category.rs)
Category theory primitives providing abstract mathematical frameworks for composition and morphisms. Enables advanced mathematical constructions and provides a unifying language for describing mathematical structures and their relationships.

## Design Principles

- **Mathematical Rigor**: All implementations follow strict mathematical definitions and maintain algebraic properties
- **Type Safety**: Leverages Rust's type system to encode mathematical constraints and prevent invalid operations
- **Composability**: Structures are designed to work together seamlessly, allowing complex mathematical constructions
- **Performance**: Balances mathematical correctness with computational efficiency through careful API design

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
cova-algebra = "*"
```

The crate provides a comprehensive prelude for convenient importing:

```rust
use cova_algebra::prelude::*;
```

## Module Hierarchy

The algebraic structures follow a natural mathematical hierarchy:

```
Arithmetic Operations
    ├── Groups (symmetry and transformation)
    ├── Rings & Fields (number systems)
    └── Modules & Vector Spaces (linear structures)
        ├── Algebras (vector spaces with multiplication)
        ├── Tensors (multi-dimensional arrays)
```

## Documentation

Complete API documentation is available on [docs.rs](https://docs.rs/cova-algebra).

## Contributing

Contributions are welcome! Please ensure mathematical correctness and include appropriate documentation for any new algebraic structures.

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](../LICENSE) file for details.

