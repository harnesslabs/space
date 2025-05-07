//! Graph implementation with support for both directed and undirected graphs.
//!
//! This module provides a flexible graph data structure that can represent both directed
//! and undirected graphs through a type parameter. The implementation supports basic
//! set operations and is designed to work with the topology traits defined in the
//! definitions module.

use std::{collections::HashSet, hash::Hash, marker::PhantomData};

use crate::{definitions::TopologicalSpace, set::Set};

/// Private module to implement the sealed trait pattern.
/// This prevents other crates from implementing DirectedType.
mod sealed {
  pub trait Sealed {}
}

/// A trait to distinguish between directed and undirected graphs.
///
/// This trait is sealed and can only be implemented by the `Directed` and
/// `Undirected` types provided in this module.
pub trait DirectedType: sealed::Sealed {
  /// Whether the graph is directed (`true`) or undirected (`false`).
  const DIRECTED: bool;
}

/// Type marker for undirected graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Undirected;
impl sealed::Sealed for Undirected {}
impl DirectedType for Undirected {
  const DIRECTED: bool = false;
}

/// Type marker for directed graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Directed;
impl sealed::Sealed for Directed {}
impl DirectedType for Directed {
  const DIRECTED: bool = true;
}

/// Represents a point in a graph, which can be either a vertex or a point on an edge.
///
/// # Type Parameters
/// * `V` - The type of vertex identifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphPoint<V> {
  /// A vertex in the graph
  Vertex(V),
  /// A point on an edge between two vertices
  EdgePoint(V, V),
}

/// A graph data structure supporting both directed and undirected graphs.
///
/// # Type Parameters
/// * `V` - The type of vertex identifiers
/// * `D` - The directedness type (`Directed` or `Undirected`)
///
/// # Examples
/// ```
/// use std::collections::HashSet;
/// # use harness_space::graph::{Graph, Undirected};
///
/// let mut vertices = HashSet::new();
/// vertices.insert(1);
/// vertices.insert(2);
///
/// let mut edges = HashSet::new();
/// edges.insert((1, 2));
///
/// let graph: Graph<_, Undirected> = Graph::new(vertices, edges);
/// ```
#[derive(Debug, Clone)]
pub struct Graph<V, D: DirectedType> {
  /// The set of vertices in the graph
  vertices: HashSet<V>,
  /// The set of edges in the graph
  edges:    HashSet<(V, V)>,
  /// Phantom data to carry the directedness type
  d:        PhantomData<D>,
}

impl<V: PartialOrd + Eq + Hash, D: DirectedType> Graph<V, D> {
  /// Creates a new graph with the given vertices and edges.
  ///
  /// For undirected graphs, edges are normalized so that the smaller vertex
  /// (by `PartialOrd`) is always first in the pair.
  ///
  /// # Arguments
  /// * `vertices` - The set of vertices in the graph
  /// * `edges` - The set of edges in the graph
  ///
  /// # Panics
  /// * If any edge references a vertex not in the vertex set
  pub fn new(vertices: HashSet<V>, edges: HashSet<(V, V)>) -> Self {
    let edges = if D::DIRECTED {
      edges
    } else {
      edges.into_iter().map(|(a, b)| if a <= b { (a, b) } else { (b, a) }).collect::<HashSet<_>>()
    };

    assert!(
      edges.iter().all(|(a, b)| vertices.contains(a) && vertices.contains(b)),
      "All edges must be between vertices",
    );
    Self { vertices, edges, d: PhantomData }
  }
}

impl<V: PartialOrd + Eq + Hash + Clone, D: DirectedType> Set for Graph<V, D> {
  type Point = GraphPoint<V>;

  /// Tests if a point is contained in the graph.
  ///
  /// # Arguments
  /// * `point` - The point to test for containment
  ///
  /// # Returns
  /// * `true` if the point is a vertex or edge point in the graph
  /// * `false` otherwise
  fn contains(&self, point: &Self::Point) -> bool {
    match point {
      GraphPoint::Vertex(v) => self.vertices.contains(v),
      GraphPoint::EdgePoint(u, v) =>
        self.edges.contains(&(u.clone(), v.clone())) | self.edges.contains(&(v.clone(), u.clone())),
    }
  }

  /// Computes the set difference of two graphs (self - other).
  ///
  /// The resulting graph contains vertices and edges that are in `self` but not in `other`.
  /// Note that edges are only included if both their vertices are in the result.
  fn difference(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.difference(&other.vertices).cloned().collect();

    let edges: HashSet<(V, V)> = self
      .edges
      .iter()
      .filter(|(u, v)| {
        self.vertices.contains(u)
          && self.vertices.contains(v)
          && !other.edges.contains(&(u.clone(), v.clone()))
      })
      .cloned()
      .collect();

    Self::new(vertices, edges)
  }

  /// Computes the intersection of two graphs.
  ///
  /// The resulting graph contains vertices and edges that are in both graphs.
  fn intersect(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.intersection(&other.vertices).cloned().collect();
    let edges: HashSet<(V, V)> = self.edges.intersection(&other.edges).cloned().collect();
    Self::new(vertices, edges)
  }

  /// Computes the union of two graphs.
  ///
  /// The resulting graph contains all vertices and edges from both graphs.
  fn union(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.union(&other.vertices).cloned().collect();
    let edges: HashSet<(V, V)> = self.edges.union(&other.edges).cloned().collect();
    Self::new(vertices, edges)
  }

  fn is_empty(&self) -> bool { self.vertices.is_empty() }
}

impl<V: PartialOrd + Eq + Hash + Clone, D: DirectedType> TopologicalSpace for Graph<V, D> {
  type OpenSet = HashSet<GraphPoint<V>>;
  type Point = GraphPoint<V>;

  fn neighborhood(&self, point: Self::Point) -> Self::OpenSet {
    let mut neighborhood = HashSet::new();
    match point {
      GraphPoint::Vertex(v) => {
        neighborhood.insert(GraphPoint::Vertex(v.clone()));
        for (u, w) in self.edges.clone().into_iter().filter(|(u, _)| *u == v) {
          neighborhood.insert(GraphPoint::EdgePoint(u, w));
        }
      },
      GraphPoint::EdgePoint(u, v) => {
        neighborhood.insert(GraphPoint::EdgePoint(u.clone(), v.clone()));
        neighborhood.insert(GraphPoint::Vertex(v));
        neighborhood.insert(GraphPoint::Vertex(u));
      },
    }
    neighborhood
  }

  fn is_open(&self, open_set: Self::OpenSet) -> bool {
    open_set.iter().all(|point| self.contains(point))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Helper function to create a test graph.
  fn create_graph() -> Graph<usize, Undirected> {
    let mut vertices = HashSet::new();
    vertices.insert(1);
    vertices.insert(2);
    vertices.insert(3);
    vertices.insert(4);
    vertices.insert(5);

    let mut edges = HashSet::new();
    edges.insert((1, 2));
    edges.insert((2, 3));
    edges.insert((3, 4));

    Graph::new(vertices, edges)
  }

  #[test]
  fn graph_builds() {
    let graph = create_graph();
    assert_eq!(graph.vertices.len(), 5);
    assert_eq!(graph.edges.len(), 3);
  }

  // TODO: Uncomment and fix these tests when implementing TopologicalSpace and MetricSpace
}
