use std::collections::{HashMap, HashSet};

use harness_algebra::{ring::Field, vector::Vector};

use crate::{
  definitions::{Set, TopologicalSpace},
  graph::{Graph, Undirected},
};

/// A trait representing a cellular sheaf over a topological space.
///
/// A sheaf assigns a vector space (stalk) to each cell and provides
/// restriction maps between stalks when cells are incident.
pub trait Presheaf<T: TopologicalSpace> {
  /// The type of the data at each point
  type Data;
  /// The type of sections over open sets
  type Section: Section<T, Stalk = Self::Data>;

  /// Restricts a section from a larger open set to a smaller one
  fn restrict(
    &self,
    section: &Self::Section,
    from: &<T as TopologicalSpace>::OpenSet,
    to: &<T as TopologicalSpace>::OpenSet,
  ) -> Self::Section;
}

pub trait Section<T: TopologicalSpace> {
  /// The type of the stalk this section takes values in
  type Stalk;

  /// Evaluates the section at a point in its domain, returning an element of the stalk
  fn evaluate(&self, point: &<T as TopologicalSpace>::Point) -> Option<Self::Stalk>;

  /// Gets the open set over which this section is defined
  fn domain(&self) -> <T as TopologicalSpace>::OpenSet;
}

/// A section over a graph that assigns vectors to vertices and edges
#[derive(Debug, Clone)]
pub struct GraphSection<V> {
  vertex_values: HashMap<V, Vec<f64>>,
  edge_values:   HashMap<(V, V), Vec<f64>>,
  domain:        Graph<V, Undirected>,
}

impl<V: Eq + std::hash::Hash + Clone> Section<Graph<V, Undirected>> for GraphSection<V> {
  type Sheaf = GraphSheaf<V>;
  type Stalk = RealVectorSpace;

  fn evaluate(&self, point: &GraphPoint<V>) -> Option<Vec<f64>> {
    match point {
      GraphPoint::Vertex(v) => self.vertex_values.get(v).cloned(),
      GraphPoint::EdgePoint(u, v) => {
        let edge = if u <= v { (u.clone(), v.clone()) } else { (v.clone(), u.clone()) };
        self.edge_values.get(&edge).cloned()
      },
    }
  }

  fn domain(&self) -> Graph<V, Undirected> { self.domain.clone() }
}

/// A cellular sheaf on a graph where vertices and edges can have different dimensional stalks
#[derive(Debug, Clone)]
pub struct GraphSheaf<V> {
  graph:                Graph<V, Undirected>,
  vertex_dimension:     usize,
  edge_dimension:       usize,
  restriction_matrices: HashMap<(V, V), Vec<Vec<f64>>>, // Maps edges to restriction matrices
}

impl<V: Eq + std::hash::Hash + Clone> Presheaf<Graph<V, Undirected>> for GraphSheaf<V> {
  type Section = GraphSection<V>;
  type Stalk = RealVectorSpace;

  fn restrict(
    &self,
    section: &Self::Section,
    from: &Graph<V, Undirected>,
    to: &Graph<V, Undirected>,
  ) -> Self::Section {
    // For cellular sheaves, restriction is just taking the subset of values
    // that correspond to the cells in the smaller graph
    let vertex_values: HashMap<_, _> = to
      .vertices
      .iter()
      .filter_map(|v| section.vertex_values.get(v).map(|val| (v.clone(), val.clone())))
      .collect();

    let edge_values: HashMap<_, _> = to
      .edges
      .iter()
      .filter_map(|(u, v)| {
        let edge = if u <= v { (u.clone(), v.clone()) } else { (v.clone(), u.clone()) };
        section.edge_values.get(&edge).map(|val| (edge, val.clone()))
      })
      .collect();

    GraphSection { vertex_values, edge_values, domain: to.clone() }
  }
}

impl<V: Eq + std::hash::Hash + Clone> GraphSheaf<V> {
  /// Creates a new graph sheaf with specified dimensions for vertex and edge stalks
  pub fn new(graph: Graph<V, Undirected>, vertex_dimension: usize, edge_dimension: usize) -> Self {
    Self { graph, vertex_dimension, edge_dimension, restriction_matrices: HashMap::new() }
  }

  /// Sets the restriction matrix for an edge
  pub fn set_restriction_matrix(
    &mut self,
    edge: (V, V),
    matrix: Vec<Vec<f64>>,
  ) -> Result<(), String> {
    // Verify matrix dimensions
    if matrix.len() != self.edge_dimension {
      return Err(format!("Matrix must have {} rows (edge dimension)", self.edge_dimension));
    }
    if matrix[0].len() != self.vertex_dimension {
      return Err(format!("Matrix must have {} columns (vertex dimension)", self.vertex_dimension));
    }

    let edge = if edge.0 <= edge.1 { edge } else { (edge.1, edge.0) };
    self.restriction_matrices.insert(edge, matrix);
    Ok(())
  }

  /// Creates a section with zero vectors for all vertices and edges
  pub fn zero_section(&self) -> GraphSection<V> {
    let vertex_values: HashMap<_, _> =
      self.graph.vertices.iter().map(|v| (v.clone(), vec![0.0; self.vertex_dimension])).collect();

    let edge_values: HashMap<_, _> = self
      .graph
      .edges
      .iter()
      .map(|(u, v)| {
        let edge = if u <= v { (u.clone(), v.clone()) } else { (v.clone(), u.clone()) };
        (edge, vec![0.0; self.edge_dimension])
      })
      .collect();

    GraphSection { vertex_values, edge_values, domain: self.graph.clone() }
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use super::*;

  fn create_test_graph() -> Graph<usize, Undirected> {
    let mut vertices = HashSet::new();
    vertices.insert(1);
    vertices.insert(2);
    vertices.insert(3);

    let mut edges = HashSet::new();
    edges.insert((1, 2));
    edges.insert((2, 3));

    Graph::new(vertices, edges)
  }

  #[test]
  fn test_zero_section() {
    let graph = create_test_graph();
    let sheaf = GraphSheaf::new(graph.clone(), 2, 1); // 2D vertex stalks, 1D edge stalks

    let section = sheaf.zero_section();

    // Check vertex values
    assert_eq!(section.vertex_values[&1], vec![0.0, 0.0]);
    assert_eq!(section.vertex_values[&2], vec![0.0, 0.0]);
    assert_eq!(section.vertex_values[&3], vec![0.0, 0.0]);

    // Check edge values
    assert_eq!(section.edge_values[&(1, 2)], vec![0.0]);
    assert_eq!(section.edge_values[&(2, 3)], vec![0.0]);
  }

  #[test]
  fn test_restriction() {
    let graph = create_test_graph();
    let mut sheaf = GraphSheaf::new(graph.clone(), 2, 1);

    // Create a subgraph with just vertices 1 and 2
    let mut sub_vertices = HashSet::new();
    sub_vertices.insert(1);
    sub_vertices.insert(2);
    let mut sub_edges = HashSet::new();
    sub_edges.insert((1, 2));
    let subgraph = Graph::new(sub_vertices, sub_edges);

    let section = sheaf.zero_section();
    let restricted = sheaf.restrict(&section, &graph, &subgraph);

    // Check that only vertices 1 and 2 and edge (1,2) are in the restricted section
    assert_eq!(restricted.vertex_values.len(), 2);
    assert_eq!(restricted.edge_values.len(), 1);
    assert!(restricted.vertex_values.contains_key(&1));
    assert!(restricted.vertex_values.contains_key(&2));
    assert!(restricted.edge_values.contains_key(&(1, 2)));
  }

  #[test]
  fn test_restriction_matrix() {
    let graph = create_test_graph();
    let mut sheaf = GraphSheaf::new(graph, 2, 1);

    // Set a restriction matrix for edge (1,2)
    let matrix = vec![vec![1.0, 0.0]]; // Projection onto first coordinate
    assert!(sheaf.set_restriction_matrix((1, 2), matrix).is_ok());

    // Try setting an invalid matrix
    let bad_matrix = vec![vec![1.0]]; // Wrong dimensions
    assert!(sheaf.set_restriction_matrix((1, 2), bad_matrix).is_err());
  }
}
