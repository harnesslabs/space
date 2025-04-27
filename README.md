# Harness

A Rust ecosystem for mathematical abstractions and computations, focusing on rigorous implementations of mathematical structures and algorithms.

## Overview

Harness provides a collection of crates that implement various mathematical structures and algorithms with a focus on type safety, correctness, and composability. The project aims to provide foundational mathematical tools that can be used in scientific computing, computer graphics, machine learning, and other domains requiring robust mathematical implementations.

## Crates

### `space`

The `space` crate implements fundamental topological and geometric structures:

- Topological spaces with rigorous axiom enforcement
- Metric spaces and distance functions
- Normed and inner product spaces
- Graph structures with topology
- Simplicial complexes and homology computations

### `algebra` (Coming Soon)

The `algebra` crate will provide implementations of algebraic structures:

- Groups, rings, and fields
- Vector spaces and linear transformations
- Modules and tensor products
- Polynomial rings and factorization

## Design Philosophy

- **Type Safety**: Mathematical properties are encoded in the type system where possible
- **Correctness**: Implementations prioritize mathematical correctness over performance
- **Composability**: Structures are designed to work together seamlessly
- **Documentation**: Extensive mathematical documentation and examples

## Getting Started

### Prerequisites

Harness requires Rust 1.70 or later.

### Installation

Add the desired crates to your `Cargo.toml`:

```toml
[dependencies]
harness-space = "0.1"
harness-algebra = "0.1"  # Coming soon
```

### Example Usage

```rust
use harness_space::{
    definitions::{MetricSpace, TopologicalSpace},
    graph::{Graph, Undirected},
};
use std::collections::HashSet;

// Create an undirected graph
let mut vertices = HashSet::new();
vertices.insert(1);
vertices.insert(2);

let mut edges = HashSet::new();
edges.insert((1, 2));

let graph: Graph<_, Undirected> = Graph::new(vertices, edges);

// Work with the graph as a metric space
let distance = graph.distance(1, 2);
```

## Contributing

We welcome contributions! Whether you're interested in:

- Adding new mathematical structures
- Improving documentation
- Adding examples and test cases
- Optimizing implementations
- Finding and fixing bugs

Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/harness.git
   cd harness
   ```

2. Run the tests:
   ```bash
   cargo test --all
   ```

## Documentation

- [API Documentation](https://docs.rs/harness-space) (Coming soon)
- [Mathematical Background](docs/math.md) (Coming soon)
- [Design Decisions](docs/design.md) (Coming soon)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

This project draws inspiration from:

- Category theory and its applications to programming
- The Haskell ecosystem's approach to mathematical abstractions
- Modern algebraic topology and its computational aspects
