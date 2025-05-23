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
  pub const fn with_id(mut self, new_id: usize) -> Self {
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
          let face = Self::new(k - 1, face_vertices);
          faces.push(face);
        }
      }
    }

    faces
  }

  fn boundary_with_orientations(&self) -> Vec<(Self, i32)> {
    if self.dimension == 0 {
      return Vec::new();
    }

    let mut faces_with_orientations = Vec::new();
    let k = self.dimension;

    // Compute cubical boundary: ∂_k σ = Σ_{i=0}^{k-1} (-1)^i (σ|_{x_i=1} - σ|_{x_i=0})
    for coord_idx in 0..k {
      // Get the two faces by fixing coordinate coord_idx to 0 and 1
      for (bit_value, base_sign) in [(0, -1), (1, 1)] {
        let mut face_vertices = Vec::new();

        // Collect vertices where the coord_idx-th bit matches bit_value
        for (vertex_idx, &vertex) in self.vertices.iter().enumerate() {
          if ((vertex_idx >> coord_idx) & 1) == bit_value {
            face_vertices.push(vertex);
          }
        }

        // Create the face with proper orientation
        if !face_vertices.is_empty() && face_vertices.len() == (1 << (k - 1)) {
          let face = Self::new(k - 1, face_vertices);
          // Cubical boundary orientation: (-1)^i * (face_1 - face_0)
          let orientation = base_sign * if coord_idx % 2 == 0 { 1 } else { -1 };
          faces_with_orientations.push((face, orientation));
        }
      }
    }

    faces_with_orientations
  }

  fn id(&self) -> Option<usize> { self.id }

  fn same_content(&self, other: &Self) -> bool { self.same_content(other) }

  fn with_id(&self, new_id: usize) -> Self { self.clone().with_id(new_id) }
}

#[cfg(test)]
mod tests {
  use harness_algebra::algebras::boolean::Boolean;

  use super::*;
  use crate::{complexes::Complex, homology::Chain};

  #[test]
  fn test_cube_creation() {
    let vertex = Cube::vertex(42);
    assert_eq!(vertex.dimension(), 0);
    assert_eq!(vertex.vertices(), &[42]);
    assert_eq!(vertex.id(), None);

    let edge = Cube::edge(10, 11);
    assert_eq!(edge.dimension(), 1);
    assert_eq!(edge.vertices(), &[10, 11]);
    assert_eq!(edge.id(), None);

    let square = Cube::square([0, 1, 2, 3]);
    assert_eq!(square.dimension(), 2);
    assert_eq!(square.vertices(), &[0, 1, 2, 3]);
    assert_eq!(square.id(), None);
  }

  #[test]
  fn test_cube_with_id() {
    let cube = Cube::edge(10, 11);
    assert_eq!(cube.id(), None);

    let cube_with_id = cube.with_id(42);
    assert_eq!(cube_with_id.id(), Some(42));
    assert_eq!(cube_with_id.vertices(), &[10, 11]); // Content unchanged
  }

  #[test]
  fn test_cube_same_content() {
    let cube1 = Cube::edge(10, 11);
    let cube2 = Cube::edge(10, 11); // Same content
    let cube3 = Cube::edge(10, 12); // Different vertices
    let cube4 = cube1.clone().with_id(42); // Same content, different ID

    assert!(cube1.same_content(&cube2));
    assert!(!cube1.same_content(&cube3));
    assert!(cube1.same_content(&cube4)); // Content equality ignores ID
  }

  #[test]
  fn test_cube_ordering() {
    let v1 = Cube::vertex(0);
    let v2 = Cube::vertex(1);
    let e1 = Cube::edge(0, 1);

    assert!(v1 < v2); // Same dimension, different vertices
    assert!(v1 < e1); // Different dimension
  }

  #[test]
  fn test_cube_faces() {
    // Test 0-cube (vertex) faces
    let vertex = Cube::vertex(42);
    let vertex_faces = vertex.faces();
    assert_eq!(vertex_faces.len(), 0); // No faces for 0-cubes

    // Test 1-cube (edge) faces
    let edge = Cube::edge(10, 11);
    let edge_faces = edge.faces();
    assert_eq!(edge_faces.len(), 2); // Two vertices

    // Both faces should be 0-cubes
    for face in &edge_faces {
      assert_eq!(face.dimension(), 0);
      assert_eq!(face.vertices().len(), 1);
    }

    // The faces should contain the edge's vertices
    let face_vertices: Vec<usize> = edge_faces.iter().flat_map(Cube::vertices).copied().collect();
    assert!(face_vertices.contains(&10));
    assert!(face_vertices.contains(&11));

    // Test 2-cube (square) faces
    let square = Cube::square([0, 1, 2, 3]);
    let square_faces = square.faces();
    assert_eq!(square_faces.len(), 4); // Four edges

    // All faces should be 1-cubes
    for face in &square_faces {
      assert_eq!(face.dimension(), 1);
      assert_eq!(face.vertices().len(), 2);
    }
  }

  #[test]
  #[should_panic = "A 1-cube must have exactly 2 vertices, got 3"]
  fn test_invalid_vertex_count() {
    // A 1-cube should have exactly 2 vertices, not 3
    Cube::new(1, vec![0, 1, 2]);
  }

  #[test]
  fn test_cubical_complex_basic() {
    let mut complex: Complex<Cube> = Complex::new();
    let square = Cube::square([0, 1, 2, 3]);
    complex.join_element(square);

    assert_eq!(complex.elements_of_dimension(2).len(), 1); // 1 square
    assert_eq!(complex.elements_of_dimension(1).len(), 4); // 4 edges
    assert_eq!(complex.elements_of_dimension(0).len(), 4); // 4 vertices
  }

  #[test]
  fn test_cubical_chain_operations() {
    let mut complex: Complex<Cube> = Complex::new();

    // Create two edges
    let edge1 = Cube::edge(0, 1);
    let edge2 = Cube::edge(1, 2);
    let added_edge1 = complex.join_element(edge1);
    let added_edge2 = complex.join_element(edge2);

    let chain1 = Chain::from_item_and_coeff(&complex, added_edge1, 1_i32);
    let chain2 = Chain::from_item_and_coeff(&complex, added_edge2, 2_i32);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.coefficients, vec![1, 2]);
  }

  #[test]
  fn test_cubical_boundary_operations() {
    let mut complex: Complex<Cube> = Complex::new();

    // Test edge boundary
    let edge = Cube::edge(0, 1);
    let added_edge = complex.join_element(edge);
    let chain = Chain::from_item_and_coeff(&complex, added_edge, 1);

    let boundary = chain.boundary();
    assert_eq!(boundary.items.len(), 2); // Two vertices

    // Test square boundary
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);
    let square_chain = Chain::from_item_and_coeff(&complex, added_square, 1);

    let square_boundary = square_chain.boundary();
    assert_eq!(square_boundary.items.len(), 4); // Four edges
  }

  #[test]
  fn test_cubical_boundary_squared_is_zero() {
    let mut complex: Complex<Cube> = Complex::new();

    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);
    let chain = Chain::from_item_and_coeff(&complex, added_square, 1);

    let boundary = chain.boundary();
    let boundary_squared = boundary.boundary();

    // Boundary of boundary should be empty (∂² = 0)
    assert_eq!(boundary_squared.items.len(), 0);
    assert_eq!(boundary_squared.coefficients.len(), 0);
  }

  #[test]
  fn test_cubical_incidence_poset_condition_1() {
    // Condition 1: If [σ : τ] ≠ 0, then σ ⊲ τ and there are no cells between σ and τ
    let mut complex: Complex<Cube> = Complex::new();

    // Add a square which will create all its faces
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);

    // Get all elements
    let vertices = complex.elements_of_dimension(0);
    let edges = complex.elements_of_dimension(1);
    let squares = complex.elements_of_dimension(2);

    // Test that square's boundary consists only of direct faces (edges)
    let boundary_with_orientations = added_square.boundary_with_orientations();
    for (face, _orientation) in boundary_with_orientations {
      // Each face should be an edge (1-dimensional)
      assert_eq!(face.dimension(), 1);
      // Each face should be in the complex
      assert!(edges.iter().any(|e| e.same_content(&face)));
      // There should be no elements between the square and its faces
      // (already guaranteed by construction since square is 2D and faces are 1D)
    }

    // Test that edges' boundaries consist only of direct faces (vertices)
    for edge in &edges {
      let edge_boundary = edge.boundary_with_orientations();
      for (vertex_face, _orientation) in edge_boundary {
        // Each face should be a vertex (0-dimensional)
        assert_eq!(vertex_face.dimension(), 0);
        // Each face should be in the complex
        assert!(vertices.iter().any(|v| v.same_content(&vertex_face)));
      }
    }
  }

  #[test]
  fn test_cubical_incidence_poset_condition_2() {
    // Condition 2: For any σ ⊲ τ, Σ_γ∈P_X [σ : γ][γ : τ] = 0 (∂² = 0)
    let mut complex: Complex<Cube> = Complex::new();

    // Test with a square
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);

    // Create a chain from the square
    let square_chain = Chain::from_item_and_coeff(&complex, added_square, 1);

    // Compute boundary of the square (should give edges)
    let boundary_1 = square_chain.boundary();

    // Compute boundary of the boundary (should be zero)
    let boundary_2 = boundary_1.boundary();

    // ∂² should be zero
    assert_eq!(boundary_2.items.len(), 0, "∂² should be zero for cubical complex");
    assert_eq!(boundary_2.coefficients.len(), 0, "∂² should have no coefficients");

    // Also test with a more complex example - a cube
    let cube = Cube::new(3, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let mut cube_complex: Complex<Cube> = Complex::new();
    let added_cube = cube_complex.join_element(cube);

    let cube_chain = Chain::from_item_and_coeff(&cube_complex, added_cube, 1);
    let cube_boundary_1 = cube_chain.boundary();
    let cube_boundary_2 = cube_boundary_1.boundary();

    assert_eq!(cube_boundary_2.items.len(), 0, "∂² should be zero for 3D cube");
  }

  #[test]
  fn test_cubical_homology_point() {
    let mut complex: Complex<Cube> = Complex::new();
    let vertex = Cube::vertex(0);
    complex.join_element(vertex);

    let h0 = complex.homology::<Boolean>(0);
    let h1 = complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 0); // No 1D holes
  }

  #[test]
  fn test_cubical_homology_edge() {
    let mut complex: Complex<Cube> = Complex::new();
    let edge = Cube::edge(0, 1);
    complex.join_element(edge);

    let h0 = complex.homology::<Boolean>(0);
    let h1 = complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 0); // No 1D holes (contractible)
  }

  #[test]
  fn test_cubical_homology_square_boundary() {
    let mut complex: Complex<Cube> = Complex::new();

    // Create a square boundary (4 edges forming a cycle)
    let edge1 = Cube::edge(0, 1);
    let edge2 = Cube::edge(1, 2);
    let edge3 = Cube::edge(2, 3);
    let edge4 = Cube::edge(3, 0);

    complex.join_element(edge1);
    complex.join_element(edge2);
    complex.join_element(edge3);
    complex.join_element(edge4);

    let h0 = complex.homology::<Boolean>(0);
    let h1 = complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 1); // One 1-dimensional hole
  }

  #[test]
  fn test_cubical_homology_filled_square() {
    let mut complex: Complex<Cube> = Complex::new();
    let square = Cube::square([0, 1, 2, 3]);
    complex.join_element(square);

    let h0 = complex.homology::<Boolean>(0);
    let h1 = complex.homology::<Boolean>(1);
    let h2 = complex.homology::<Boolean>(2);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 0); // No 1D holes (filled)
    assert_eq!(h2.betti_number, 0); // No 2D holes
  }

  #[test]
  fn test_cubical_homology_two_disjoint_squares() {
    let mut complex: Complex<Cube> = Complex::new();

    let square1 = Cube::square([0, 1, 2, 3]);
    let square2 = Cube::square([4, 5, 6, 7]);
    complex.join_element(square1);
    complex.join_element(square2);

    let h0 = complex.homology::<Boolean>(0);
    let h1 = complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 2); // Two connected components
    assert_eq!(h1.betti_number, 0); // No 1D holes (both filled)
  }
}
