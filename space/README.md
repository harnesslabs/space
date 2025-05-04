[![Crates.io - harness-space](https://img.shields.io/crates/v/harness-space?label=harness-space)](https://crates.io/crates/harness-space)
[![docs.rs - harness-space](https://img.shields.io/docsrs/harness-space?label=docs.rs%20harness-space)](https://docs.rs/harness-space)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Space Crate

A Rust library providing mathematical structures and operations for topological spaces, simplicial complexes, and graphs. This crate is designed to support computational topology and geometry applications.

## Features

- **Topological Spaces**: Implementation of fundamental topological concepts
  - Sets with basic operations (union, intersection, difference)
  - Topological spaces with neighborhoods and open sets
  - Metric spaces with distance functions
  - Normed and inner product spaces

- **Simplicial Complexes**: Tools for working with simplicial complexes
  - Simplex representation (points, edges, triangles, etc.)
  - Chain complexes with boundary operations
  - Support for arbitrary coefficient rings

- **Graph Theory**: Flexible graph data structures
  - Support for both directed and undirected graphs
  - Basic graph operations and set operations
  - Vertex and edge point representation

## Usage

### Topological Spaces

```rust, ignore
use harness_space::definitions::{Set, TopologicalSpace, MetricSpace};

// Define your own space type
struct MySpace {
    // ... implementation details
}

struct MyPoint {
    // ... implementation details
}

impl Set for MySpace {
    type Point = MyPoint;
    // ... implement set operations
}

impl TopologicalSpace for MySpace {
    type Point = MyPoint;
    type OpenSet = MyOpenSet;
    // ... implement topological operations
}
```

### Simplicial Complexes

```rust
use harness_space::simplicial::{Simplex, SimplicialComplex, Chain};

// Create a simplex (e.g., a triangle)
let triangle = Simplex::new(2, vec![0, 1, 2]);

// Create a simplicial complex
let mut complex = SimplicialComplex::new();
complex.join_simplex(triangle);

// Compute boundaries
let boundary = complex.boundary::<i32>(2);
```

### Graphs

```rust
use harness_space::graph::{Graph, Undirected};
use std::collections::HashSet;

// Create a graph
let mut vertices = HashSet::new();
vertices.insert(1);
vertices.insert(2);

let mut edges = HashSet::new();
edges.insert((1, 2));

let graph: Graph<_, Undirected> = Graph::new(vertices, edges);
```

## Dependencies

- `itertools`: For combinatorial operations
- `num`: For numeric traits and operations

## Examples

### Creating a Simplicial Complex

```rust
use harness_space::simplicial::{Simplex, SimplicialComplex};

// Create a tetrahedron
let mut complex = SimplicialComplex::new();
complex.join_simplex(Simplex::new(3, vec![0, 1, 2, 3]));
```

### Working with Graphs

```rust
use harness_space::graph::{Graph, Directed};
use std::collections::HashSet;

// Create a directed graph
let vertices: HashSet<_> = [1, 2, 3].into_iter().collect();
let edges: HashSet<_> = [(1, 2), (2, 3)].into_iter().collect();
let graph: Graph<_, Directed> = Graph::new(vertices, edges);
```


