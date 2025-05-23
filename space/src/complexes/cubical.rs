//! # Cubical Complexes and Cubical Topology
//!
//! This module provides data structures for representing **cubical complexes**, which are
//! fundamental objects in algebraic topology and computational topology. Cubical complexes
//! are topological spaces built by attaching $k$-dimensional cubes (hypercubes) together.
//!
//! ## Core Components
//!
//! - **[`Cube`]**: Represents an individual $k$-cube. Each cube has a dimension and is defined by
//!   its vertices in a discrete grid structure.
//!
//! ## Mathematical Background
//!
//! A $k$-dimensional cube (or $k$-cube) in this implementation is represented as a collection
//! of $2^k$ vertices that form the corners of a hypercube. For example:
//! - A $0$-cube is a point (1 vertex)
//! - A $1$-cube is a unit edge (2 vertices)
//! - A $2$-cube is a unit square (4 vertices)
//! - A $3$-cube is a unit cube in 3D space (8 vertices)
//!
//! The faces of a $k$-cube are obtained by systematic selection of vertex subsets that
//! correspond to $(k-1)$-dimensional faces of the hypercube.

use std::fmt;

use super::ComplexElement;

/// Represents an individual cube in a cubical complex.
///
/// A $k$-cube is represented by its $2^k$ vertices and its dimension.
/// The vertices are ordered according to a binary coordinate system where
/// each vertex corresponds to a corner of the hypercube.
///
/// # Fields
///
/// * `vertices`: A `Vec<usize>` containing the vertex indices that define the cube. For a $k$-cube,
///   this contains $2^k$ vertices ordered systematically.
/// * `dimension`: The intrinsic dimension of the cube (e.g., 0 for a point, 1 for an edge).
/// * `id`: An optional unique identifier assigned when the cube is added to a complex.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cube {
  /// The vertex indices that define this cube.
  pub vertices:  Vec<usize>,
  /// The dimension of this cube (e.g., 0 for a point, 1 for an edge, 2 for a face).
  pub dimension: usize,
  /// An optional unique identifier assigned when the cube is added to a complex.
  pub id:        Option<usize>,
}

impl Cube {
  /// Creates a new cube with the given dimension and vertices.
  ///
  /// # Arguments
  ///
  /// * `dimension`: The dimension of the cube.
  /// * `vertices`: The vertex indices that define the cube. For a k-cube, this should contain
  ///   exactly 2^k vertices.
  ///
  /// # Panics
  ///
  /// Panics if `vertices.len()` is not equal to `2^dimension`.
  pub fn new(dimension: usize, vertices: Vec<usize>) -> Self {
    let expected_vertex_count = 1 << dimension; // 2^dimension
    assert_eq!(
      vertices.len(),
      expected_vertex_count,
      "A {}-cube must have exactly {} vertices, got {}",
      dimension,
      expected_vertex_count,
      vertices.len()
    );
    Self { vertices, dimension, id: None }
  }

  /// Creates a new 0-cube (vertex) from a single vertex index.
  pub fn vertex(vertex: usize) -> Self { Self::new(0, vec![vertex]) }

  /// Creates a new 1-cube (edge) from two vertex indices.
  pub fn edge(v0: usize, v1: usize) -> Self { Self::new(1, vec![v0, v1]) }

  /// Creates a new 2-cube (square) from four vertex indices.
  /// The vertices should be ordered as: [bottom-left, bottom-right, top-left, top-right]
  /// corresponding to binary coordinates (0,0), (1,0), (0,1), (1,1).
  pub fn square(vertices: [usize; 4]) -> Self { Self::new(2, vertices.to_vec()) }

  /// Creates a new cube with a specific ID.
  pub fn with_id(mut self, new_id: usize) -> Self {
    self.id = Some(new_id);
    self
  }

  /// Returns a slice reference to the vertex indices of the cube.
  pub fn vertices(&self) -> &[usize] { &self.vertices }

  /// Returns the dimension of the cube.
  pub const fn dimension(&self) -> usize { self.dimension }

  /// Returns the ID of the cube if it has been assigned to a complex.
  pub const fn id(&self) -> Option<usize> { self.id }

  /// Checks if this cube has the same mathematical content as another.
  pub fn same_content(&self, other: &Self) -> bool {
    self.dimension == other.dimension && self.vertices == other.vertices
  }
}

impl PartialOrd for Cube {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Cube {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // First compare by dimension, then by vertices
    self.dimension.cmp(&other.dimension).then_with(|| self.vertices.cmp(&other.vertices))
  }
}

impl fmt::Display for Cube {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Cube{}(vertices:{:?})", self.dimension, self.vertices)
  }
}

impl ComplexElement for Cube {
  fn dimension(&self) -> usize { self.dimension }

  fn faces(&self) -> Vec<Self> {
    if self.dimension == 0 {
      return Vec::new(); // 0-cubes have no faces
    }

    let mut faces = Vec::new();
    let k = self.dimension;

    // For a k-cube, faces are (k-1)-cubes obtained by fixing one coordinate
    // In the binary representation, this means fixing one bit position
    for coord_idx in 0..k {
      // For each coordinate direction, create faces by fixing that coordinate
      // to both 0 and 1
      for bit_value in [0, 1] {
        let mut face_vertices = Vec::new();

        // Collect vertices where the coord_idx-th bit matches bit_value
        for (vertex_idx, &vertex) in self.vertices.iter().enumerate() {
          if ((vertex_idx >> coord_idx) & 1) == bit_value {
            face_vertices.push(vertex);
          }
        }

        // Create the (k-1)-dimensional face
        if !face_vertices.is_empty() && face_vertices.len() == (1 << (k - 1)) {
          let face = Cube::new(k - 1, face_vertices);
          faces.push(face);
        }
      }
    }

    faces
  }

  fn id(&self) -> Option<usize> { self.id }

  fn same_content(&self, other: &Self) -> bool { self.same_content(other) }

  fn with_id(&self, new_id: usize) -> Self { self.clone().with_id(new_id) }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_cube_creation() {
    let vertex = Cube::vertex(42);
    assert_eq!(vertex.dimension(), 0);
    assert_eq!(vertex.vertices(), &[42]);

    let edge = Cube::edge(10, 11);
    assert_eq!(edge.dimension(), 1);
    assert_eq!(edge.vertices(), &[10, 11]);

    let square = Cube::square([0, 1, 2, 3]);
    assert_eq!(square.dimension(), 2);
    assert_eq!(square.vertices(), &[0, 1, 2, 3]);
  }

  #[test]
  fn test_cube_faces() {
    // Test 1-cube (edge) faces
    let edge = Cube::edge(10, 11);
    let faces = edge.faces();
    assert_eq!(faces.len(), 2);

    // Both faces should be 0-cubes
    for face in &faces {
      assert_eq!(face.dimension(), 0);
      assert_eq!(face.vertices().len(), 1);
    }

    // The faces should contain the edge's vertices
    let face_vertices: Vec<usize> = faces.iter().flat_map(|f| f.vertices()).copied().collect();
    assert!(face_vertices.contains(&10));
    assert!(face_vertices.contains(&11));
  }

  #[test]
  fn test_cube_same_content() {
    let cube1 = Cube::edge(10, 11);
    let cube2 = Cube::edge(10, 11); // Same content
    let cube3 = Cube::edge(10, 12); // Different vertices

    assert!(cube1.same_content(&cube2));
    assert!(!cube1.same_content(&cube3));
  }

  #[test]
  #[should_panic]
  fn test_invalid_vertex_count() {
    // A 1-cube should have exactly 2 vertices, not 3
    Cube::new(1, vec![0, 1, 2]);
  }
}
