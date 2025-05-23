# Cova

A unified Rust ecosystem for rigorous mathematical computation, bridging abstract algebra and computational topology to provide a comprehensive foundation for mathematical software development.

[![Crates.io - cova](https://img.shields.io/crates/v/cova?label=cova)](https://crates.io/crates/cova)
[![docs.rs - cova](https://img.shields.io/docsrs/cova?label=docs.rs%20cova)](https://docs.rs/cova)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Mathematical Vision

Cova represents a principled approach to mathematical software, where abstract algebraic structures provide the computational foundation for sophisticated topological and geometric algorithms. By unifying algebra and topology in a single ecosystem, Cova enables mathematical computations that span traditional disciplinary boundaries.

## Unified Architecture

The Cova ecosystem consists of two complementary mathematical domains:

### Abstract Algebra (`cova-algebra`)
Provides the algebraic foundation with rigorous implementations of:
- **Arithmetic Systems**: Modular arithmetic, ring theory, and field extensions
- **Structural Algebra**: Groups, rings, fields, modules, and vector spaces  
- **Advanced Constructions**: Category theory, tensor algebra, and Clifford algebras
- **Computational Framework**: Type-safe operations preserving mathematical properties

### Computational Topology (`cova-space`)
Builds sophisticated topological structures and algorithms:
- **Foundational Topology**: Topological spaces, metric structures, and geometric abstractions
- **Complex Machinery**: Simplicial and cubical complexes with efficient storage and operations
- **Homological Algebra**: Chain complexes, boundary operators, and homology computation
- **Advanced Methods**: Sheaf theory, persistent homology, and topological data analysis

## Algebraic-Topological Synergy

The power of Cova emerges from the interaction between its algebraic and topological components:

### Coefficient Systems
Topological computations (homology, cohomology) use algebraic structures as coefficient systems, allowing homology over arbitrary rings and fields defined in `cova-algebra`.

### Geometric Algebra
Clifford algebras from `cova-algebra` provide natural frameworks for geometric computations in the topological spaces of `cova-space`.

### Categorical Foundations
Category theory concepts span both domains, providing unified abstractions for mathematical constructions and morphisms.

### Computational Efficiency
Algebraic optimizations (efficient field arithmetic, tensor operations) directly enhance topological algorithms that depend on large-scale linear algebra.

## Usage

### Unified Access
Import the entire ecosystem:
```rust
use cova::prelude::*;
```

### Domain-Specific Access
Access individual mathematical domains:
```rust
use cova::algebra::prelude::*;  // Abstract algebra
use cova::space::prelude::*;    // Computational topology
```

### Cross-Domain Computations
Leverage both domains together:
```rust
use cova::prelude::*;

// Compute homology over a custom finite field
let field_element = Mod7::new(3);
let homology = complex.homology::<Mod7>(dimension);

// Use Clifford algebra for geometric transformations
let rotor = CliffordElement::from_bivector(bivector);
let transformed_space = apply_rotor(space, rotor);
```

## Mathematical Scope

Cova covers essential mathematical areas with seamless integration:

```
Cova Ecosystem
├── Abstract Algebra
│   ├── Arithmetic (modular, primitive types)
│   ├── Group Theory (abelian, non-abelian)
│   ├── Ring Theory (rings, fields, semirings)
│   ├── Module Theory (vector spaces, linear algebra)
│   ├── Advanced Algebra (Clifford, Boolean)
│   ├── Tensor Calculus (fixed, dynamic tensors)
│   └── Category Theory (morphisms, composition)
└── Computational Topology  
    ├── Set Theory (collections, posets)
    ├── Topological Spaces (metric, normed)
    ├── Cell Complexes (simplicial, cubical)
    ├── Homological Algebra (chains, homology)
    ├── Sheaf Theory (categorical constructions)
    └── Topological Data Analysis (filtrations)
```

## Design Principles

### Mathematical Rigor
All implementations follow strict mathematical definitions with proper algebraic and topological properties preserved through the type system.

### Compositional Architecture  
Structures are designed for seamless composition, allowing complex mathematical constructions from fundamental building blocks across both algebraic and topological domains.

### Performance Through Abstraction
High-level mathematical abstractions are implemented with careful attention to computational efficiency, enabling both correctness and performance.

### Type-Driven Safety
Rust's type system encodes mathematical constraints, preventing invalid operations while maintaining zero-cost abstractions.

## Feature Integration

### Optional Features
- **`parallel`**: Enables parallel computation acros topological algorithms (filtration construction)

### Cross-Crate Compatibility
All types and traits are designed for interoperability, allowing algebraic structures to serve as coefficient systems for topological computations and vice versa.

## Applications

Cova provides a foundation for:
- **Scientific Computing**: Rigorous mathematical foundations for numerical algorithms
- **Computer Graphics**: Geometric algebra and topological methods for rendering and modeling  
- **Machine Learning**: Topological data analysis and algebraic machine learning methods
- **Mathematical Research**: Experimental mathematics with computational verification
- **Educational Software**: Teaching tools with mathematically accurate implementations

## Documentation

Complete documentation covering both mathematical foundations and practical usage:
- [Unified API Documentation](https://docs.rs/cova)
- [Algebra Documentation](https://docs.rs/cova-algebra) 
- [Topology Documentation](https://docs.rs/cova-space)

## License

This project is licensed under the AGPLv3 License, ensuring mathematical software remains open and accessible to the research community. 