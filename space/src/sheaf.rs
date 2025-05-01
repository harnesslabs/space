use std::collections::{HashMap, HashSet};

use harness_algebra::{
  ring::Field,
  vector::{Vector, VectorSpace},
};

use crate::{
  definitions::TopologicalSpace,
  graph::{Graph, GraphPoint, Undirected},
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

impl Section<Graph<Undirected>> for HashMap<GraphPoint, Vector<3, f64>> {
  type Stalk = Vector<3, f64>;

  fn evaluate(&self, point: &GraphPoint) -> Option<Vector<3, f64>> { self.get(point).copied() }

  fn domain(&self) -> HashSet<GraphPoint> { self.keys().cloned().collect() }
}

/// A cellular sheaf on a graph where vertices and edges can have different dimensional stalks
#[derive(Debug, Clone)]
pub struct GraphSheaf<V> {
  graph:                Graph<Undirected>,
  vertex_dimension:     usize,
  edge_dimension:       usize,
  restriction_matrices: HashMap<(V, V), Vec<Vec<f64>>>, // Maps edges to restriction matrices
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
