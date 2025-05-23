//! Graph implementation with support for both directed and undirected graphs.
//!
//! This module provides a flexible graph data structure that can represent both directed
//! and undirected graphs through a type parameter. The implementation supports basic
//! set operations and is designed to work with the topology traits defined in the
//! definitions module.

use std::{collections::HashSet, hash::Hash, marker::PhantomData};

use crate::set::Collection;

/// Private module to implement the sealed trait pattern.
/// This prevents other crates from implementing DirectedType.
mod sealed {
  pub trait Sealed {}
}

/// A trait to distinguish between directed and undirected graphs.
///
/// This trait is sealed and can only be implemented by the `Directed` and
/// `Undirected` types provided in this module.
pub trait DirectedType: sealed::Sealed {}

/// Type marker for undirected graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Undirected;
impl sealed::Sealed for Undirected {}
impl DirectedType for Undirected {}

/// Type marker for directed graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Directed;
impl sealed::Sealed for Directed {}
impl DirectedType for Directed {}

/// Represents a point in a graph, which can be either a vertex or a point on an edge.
///
/// # Type Parameters
/// * `V` - The type of vertex identifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VertexOrEdge<V> {
  /// A vertex in the graph
  Vertex(V),
  /// An edge between two vertices
  Edge(V, V),
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
/// # use cova_space::graph::{Graph, Undirected};
///
/// let mut vertices = HashSet::new();
/// vertices.insert(1);
/// vertices.insert(2);
///
/// let mut edges = HashSet::new();
/// edges.insert((1, 2));
///
/// let graph = Graph::<usize, Undirected>::new(vertices, edges);
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

impl<V: PartialOrd + Eq + Hash> Graph<V, Undirected> {
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
    let edges =
      edges.into_iter().map(|(a, b)| if a <= b { (a, b) } else { (b, a) }).collect::<HashSet<_>>();

    assert!(
      edges.iter().all(|(a, b)| vertices.contains(a) && vertices.contains(b)),
      "All edges must be between vertices",
    );
    Self { vertices, edges, d: PhantomData }
  }
}

impl<V: PartialOrd + Eq + Hash> Graph<V, Directed> {
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
    assert!(
      edges.iter().all(|(a, b)| vertices.contains(a) && vertices.contains(b)),
      "All edges must be between vertices",
    );
    Self { vertices, edges, d: PhantomData }
  }
}

impl<V: PartialOrd + Eq + Hash + Clone> Collection for Graph<V, Directed> {
  type Item = VertexOrEdge<V>;

  fn is_empty(&self) -> bool { self.vertices.is_empty() }

  fn contains(&self, point: &Self::Item) -> bool {
    match point {
      VertexOrEdge::Vertex(v) => self.vertices.contains(v),
      VertexOrEdge::Edge(u, v) => self.edges.contains(&(u.clone(), v.clone())),
    }
  }
}

impl<V: PartialOrd + Eq + Hash + Clone> Collection for Graph<V, Undirected> {
  type Item = VertexOrEdge<V>;

  fn is_empty(&self) -> bool { self.vertices.is_empty() }

  fn contains(&self, point: &Self::Item) -> bool {
    match point {
      VertexOrEdge::Vertex(v) => self.vertices.contains(v),
      VertexOrEdge::Edge(u, v) =>
        self.edges.contains(&(u.clone(), v.clone())) | self.edges.contains(&(v.clone(), u.clone())),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Helper function to create a test graph.
  fn create_graph_undirected() -> Graph<usize, Undirected> {
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

    Graph::<usize, Undirected>::new(vertices, edges)
  }

  /// Helper function to create a test graph.
  fn create_graph_directed() -> Graph<usize, Directed> {
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
    edges.insert((4, 5));

    Graph::<usize, Directed>::new(vertices, edges)
  }

  #[test]
  fn graph_builds_undirected() {
    let graph = create_graph_undirected();
    assert_eq!(graph.vertices.len(), 5);
    assert_eq!(graph.edges.len(), 3);
  }

  #[test]
  fn graph_builds_directed() {
    let graph = create_graph_directed();
    assert_eq!(graph.vertices.len(), 5);
    assert_eq!(graph.edges.len(), 4);
  }

  #[test]
  fn graph_contains_vertex() {
    let graph = create_graph_undirected();
    assert!(graph.contains(&VertexOrEdge::Vertex(1)));
    assert!(!graph.contains(&VertexOrEdge::Vertex(6)));
  }

  #[test]
  fn graph_contains_edge() {
    let graph = create_graph_undirected();
    assert!(graph.contains(&VertexOrEdge::Edge(1, 2)));
  }

  #[test]
  fn graph_contains_edge_undirected() {
    let graph = create_graph_undirected();
    assert!(graph.contains(&VertexOrEdge::Edge(1, 2)));
    assert!(graph.contains(&VertexOrEdge::Edge(2, 1)));
  }

  #[test]
  fn graph_contains_edge_directed() {
    let graph = create_graph_directed();
    assert!(graph.contains(&VertexOrEdge::Edge(1, 2)));
    assert!(!graph.contains(&VertexOrEdge::Edge(2, 1)));
  }
}
