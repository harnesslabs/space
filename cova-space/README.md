<p align="center">
  <img src="https://raw.githubusercontent.com/harnesslabs/brand/main/cova/cova.png" alt="Cova Logo" width="400">
</p>

# Cova Space

A comprehensive Rust library for computational topology and geometric analysis, providing rigorous implementations of topological spaces, simplicial complexes, homology computation, and topological data analysis.

[![Crates.io - cova-space](https://img.shields.io/crates/v/cova-space?label=cova-space)](https://crates.io/crates/cova-space)
[![docs.rs - cova-space](https://img.shields.io/docsrs/cova-space?label=docs.rs%20cova-space)](https://docs.rs/cova-space)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Overview

Cova Space implements fundamental structures and algorithms from computational topology with a focus on mathematical rigor, type safety, and performance. The crate provides a comprehensive toolkit for topological computation, from basic set operations to advanced persistent homology and sheaf-theoretic constructions.

## Architecture

The library is organized around core topological concepts, building from foundational structures to sophisticated computational tools:

### Core Foundations

#### [`set`](src/set.rs)
Foundation layer providing collection abstractions and partially ordered sets (posets). Implements basic set operations, ordering relationships, and the mathematical framework for more complex topological structures.

#### [`definitions`](src/definitions.rs)
Fundamental trait hierarchy for mathematical spaces including topological spaces, metric spaces, normed spaces, and inner product spaces. Establishes the interface for geometric and topological operations with proper mathematical abstractions.

### Topological Complexes

#### [`complexes`](src/complexes/)
Comprehensive implementation of cell complexes including simplicial and cubical complexes. Provides generic complex containers, automatic face relation management, and efficient storage with ID-based lattice structures for computational topology applications.

**Submodules:**
- **`simplicial`**: Simplex definitions and simplicial complex operations
- **`cubical`**: Cube definitions and cubical complex operations

#### [`graph`](src/graph.rs)
Flexible graph data structures supporting both directed and undirected graphs with comprehensive operations for vertices, edges, and topological relationships. Designed for integration with complex and homological computations.

### Computational Topology

#### [`homology`](src/homology.rs)
Complete homology computation framework including chain complexes, boundary operators, and Betti number calculations. Implements formal chains with ring coefficients and supports homology computation over arbitrary fields for topological analysis.

#### [`sheaf`](src/sheaf.rs)
Advanced sheaf theory implementations providing categorical constructions over topological spaces. Includes restriction morphisms, global section verification, and coboundary operators for sophisticated topological data analysis.

### Topological Data Analysis

#### [`filtration`](src/filtration/)
Filtration frameworks for persistent homology including Vietoris-Rips constructions. Supports both serial and parallel computation of filtered complexes for analyzing multi-scale topological features in data.

#### [`cloud`](src/cloud.rs)
Point cloud analysis tools designed for topological data analysis applications. Provides the foundation for building filtered complexes from geometric data sets.

#### [`lattice`](src/lattice.rs)
Sophisticated lattice structures for efficient representation of partial orders and face relationships in complexes. Implements join/meet operations and provides the computational backbone for complex operations.

## Design Principles

- **Mathematical Rigor**: All implementations follow strict topological definitions and maintain structural invariants
- **Type Safety**: Leverages Rust's type system to encode topological properties and prevent invalid operations  
- **Computational Efficiency**: Optimized data structures and algorithms for large-scale topological computations
- **Composability**: Modular design allows complex topological constructions from fundamental building blocks

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
cova-space = "*"
```

The crate provides a comprehensive prelude for convenient importing:

```rust
use cova_space::prelude::*;
```

## Feature Highlights

- **Generic Complex Framework**: Unified interface for simplicial, cubical, and general cell complexes
- **Homology Computation**: Full chain complex machinery with boundary operators and Betti number calculation
- **Persistent Homology**: Filtration frameworks for multi-scale topological analysis
- **Sheaf Theory**: Advanced categorical constructions for topological data analysis
- **High Performance**: Efficient lattice-based storage and optional parallel computation support

## Optional Features

- **`parallel`**: Enables parallel computation for filtrations and large-scale operations using Rayon

## Mathematical Scope

The library covers essential areas of computational topology:

```
Set Theory & Posets
    ├── Topological Spaces (metric, normed, inner product)
    ├── Cell Complexes (simplicial, cubical, general)
    ├── Homological Algebra (chains, boundaries, homology)
    ├── Sheaf Theory (categorical constructions)
    └── Topological Data Analysis (filtrations, persistence)
```

## Documentation

Complete API documentation is available on [docs.rs](https://docs.rs/cova-space).

## Contributing

Contributions are welcome! Please ensure mathematical correctness and include appropriate documentation for topological algorithms and data structures.

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](../LICENSE) file for details.


