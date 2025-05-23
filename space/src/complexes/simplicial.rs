use itertools::Itertools;

use super::*;

/// A simplex represents a $k$-dimensional geometric object, defined as the convex hull of $k+1$
/// affinely independent vertices.
///
/// Simplices are the basic building blocks of [`SimplicialComplex`]es.
/// - A 0-simplex (dimension 0) is a point.
/// - A 1-simplex (dimension 1) is a line segment.
/// - A 2-simplex (dimension 2) is a triangle.
/// - A 3-simplex (dimension 3) is a tetrahedron.
/// - ... and so on for higher dimensions.
///
/// In this implementation, vertices are represented by `usize` indices and are always stored in a
/// sorted vector to ensure a canonical representation for each simplex.
///
/// # Fields
/// * `vertices`: A sorted `Vec<usize>` of vertex indices that define the simplex.
/// * `dimension`: The dimension of the simplex, equal to `vertices.len() - 1`.
/// * `id`: An optional unique identifier assigned when the simplex is added to a complex.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Simplex {
  vertices:  Vec<usize>,
  dimension: usize,
  id:        Option<usize>,
}

impl Eq for Simplex {}

impl PartialOrd for Simplex {
  /// Provides a partial ordering for simplices.
  ///
  /// This implementation delegates to `cmp`, thus providing a total ordering.
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Simplex {
  /// Provides a total ordering for simplices, primarily for use in sorted collections (e.g.,
  /// `BTreeSet` or when sorting `Vec<Simplex>`).
  ///
  /// The ordering is based on the lexicographical comparison of their sorted vertex lists,
  /// then by dimension.
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.vertices.cmp(&other.vertices).then_with(|| self.dimension.cmp(&other.dimension))
  }
}

impl Simplex {
  /// Creates a new simplex of a given `dimension` from a set of `vertices`.
  ///
  /// The provided `vertices` will be sorted internally to ensure a canonical representation.
  /// The simplex is created without an ID (id = None).
  ///
  /// # Arguments
  /// * `dimension`: The dimension of the simplex (e.g., 0 for a point, 1 for an edge, 2 for a
  ///   triangle).
  /// * `vertices`: A vector of `usize` vertex indices. The length of this vector must be `dimension
  ///   + 1`.
  ///
  /// # Panics
  /// * If `vertices.len()` does not equal `dimension + 1`.
  /// * If any vertex indices in `vertices` are repeated (i.e., vertices are not distinct).
  pub fn new(dimension: usize, vertices: Vec<usize>) -> Self {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices: vertices.into_iter().sorted().collect(), dimension, id: None }
  }

  /// Creates a new simplex from vertices, automatically determining the dimension.
  /// This is useful when you want to create a simplex without specifying the dimension explicitly.
  pub fn from_vertices(vertices: Vec<usize>) -> Self {
    let dimension = vertices.len().saturating_sub(1);
    Self::new(dimension, vertices)
  }

  /// Creates a new simplex with a specific ID.
  pub fn with_id(mut self, new_id: usize) -> Self {
    self.id = Some(new_id);
    self
  }

  /// Returns a slice reference to the sorted vertex indices of the simplex.
  pub fn vertices(&self) -> &[usize] { &self.vertices }

  /// Returns the dimension of the simplex.
  ///
  /// The dimension $k$ is equal to the number of vertices minus one.
  pub const fn dimension(&self) -> usize { self.dimension }

  /// Returns the ID of the simplex if it has been assigned to a complex.
  pub const fn id(&self) -> Option<usize> { self.id }

  /// Checks if this simplex has the same mathematical content as another.
  pub fn same_content(&self, other: &Self) -> bool {
    self.dimension == other.dimension && self.vertices == other.vertices
  }
}

impl ComplexElement for Simplex {
  fn dimension(&self) -> usize { self.dimension }

  /// Computes all $(k-1)$-dimensional faces of this $k$-simplex.
  ///
  /// A face is obtained by removing one vertex from the simplex's vertex set.
  /// For example:
  /// - The faces of a 2-simplex (triangle) `[v_0, v_1, v_2]` are its three 1-simplices (edges):
  ///   `[v_1, v_2]`, `[v_0, v_2]`, and `[v_0, v_1]`.
  /// - The faces of a 1-simplex (edge) `[v_0, v_1]` are its two 0-simplices (vertices): `[v_1]` and
  ///   `[v_0]`.
  /// - A 0-simplex has no faces (its dimension is -1, typically considered an empty set of faces).
  ///
  /// # Returns
  /// A [`Vec<Simplex>`] containing all $(k-1)$-dimensional faces. If the simplex is
  /// 0-dimensional, an empty vector is returned as it has no $( -1)$-dimensional faces in the
  /// typical sense.
  fn faces(&self) -> Vec<Self> {
    if self.dimension == 0 {
      return Vec::new();
    }
    self
      .vertices
      .clone()
      .into_iter()
      .combinations(self.dimension)
      .map(|v| Self::from_vertices(v))
      .collect()
  }

  fn id(&self) -> Option<usize> { self.id }

  fn same_content(&self, other: &Self) -> bool { self.same_content(other) }

  fn with_id(&self, new_id: usize) -> Self { self.clone().with_id(new_id) }
}

#[cfg(test)]
mod tests {
  // TODO: Verify the homology generators are correct.

  use std::fmt::Debug;

  use harness_algebra::{algebras::boolean::Boolean, modular, prime_field, rings::Field};

  use super::*;

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);
  // Helper trait bound alias for tests
  trait TestField: Field + Copy + Debug {}
  impl<T: Field + Copy + Debug> TestField for T {}

  #[test]
  fn test_simplex_faces() {
    let simplex = Simplex::new(2, vec![0, 1, 2]);
    let faces = simplex.faces();
    assert_eq!(faces.len(), 3);
    assert_eq!(faces[0].vertices(), &[0, 1]);
    assert_eq!(faces[1].vertices(), &[0, 2]);
    assert_eq!(faces[2].vertices(), &[1, 2]);
  }

  #[test]
  fn test_simplicial_complex() {
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(Simplex::new(2, vec![0, 1, 2]));
    assert_eq!(complex.simplices_by_dimension(2).unwrap().len(), 1);
    assert_eq!(complex.simplices_by_dimension(1).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(0).unwrap().len(), 3);
  }

  #[test]
  fn test_chain_addition_disjoint() {
    let mut complex = SimplicialComplex::new();
    // Create two chains with different simplices
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![1, 2]);
    complex.join_simplex(simplex1.clone());
    complex.join_simplex(simplex2.clone());

    let chain1 = Chain::from_item_and_coeff(&complex, simplex1, 1_i32);
    let chain2 = Chain::from_item_and_coeff(&complex, simplex2, 2_i32);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items[0].vertices(), &[0, 1]);
    assert_eq!(result.items[1].vertices(), &[1, 2]);
    assert_eq!(result.coefficients[0], 1);
    assert_eq!(result.coefficients[1], 2);
  }

  #[test]
  fn test_chain_addition_same_simplex() {
    // Create two chains with the same simplex
    let mut complex = SimplicialComplex::new();
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(simplex1.clone());
    complex.join_simplex(simplex2.clone());

    let chain1 = Chain::from_item_and_coeff(&complex, simplex1, 2);
    let chain2 = Chain::from_item_and_coeff(&complex, simplex2, 3);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.coefficients.len(), 1);
    assert_eq!(result.items[0].vertices(), &[0, 1]);
    assert_eq!(result.coefficients[0], 5); // 2 + 3 = 5
  }

  #[test]
  fn test_chain_addition_canceling_coefficients() {
    // Create two chains with the same simplex but opposite coefficients
    let mut complex = SimplicialComplex::new();
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(simplex1.clone());
    complex.join_simplex(simplex2.clone());

    let chain1 = Chain::from_item_and_coeff(&complex, simplex1, 2);
    let chain2 = Chain::from_item_and_coeff(&complex, simplex2, -2);

    let result = chain1 + chain2;

    // The result should be empty since the coefficients cancel out
    assert_eq!(result.items.len(), 0);
    assert_eq!(result.coefficients.len(), 0);
  }

  #[test]
  fn test_chain_boundary_edge() {
    // The boundary of an edge is its two vertices with opposite signs
    let mut complex = SimplicialComplex::new();
    let edge = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(edge.clone());
    let chain = Chain::from_item_and_coeff(&complex, edge, 1);

    let boundary = chain.boundary();

    // Should have two 0-simplices (vertices)
    assert_eq!(boundary.items.len(), 2);
    assert_eq!(boundary.coefficients.len(), 2);

    // Verify the vertices
    assert!(boundary.items.iter().any(|s| s.vertices().contains(&0)));
    assert!(boundary.items.iter().any(|s| s.vertices().contains(&1)));
    assert!(boundary.items.len() == 2);

    // Verify opposite signs (exact sign depends on your implementation)
    assert_eq!(boundary.coefficients[0] + boundary.coefficients[1], 0);
  }

  #[test]
  fn test_chain_boundary_triangle() {
    // The boundary of a triangle is its three edges
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(triangle.clone());
    let chain = Chain::from_item_and_coeff(&complex, triangle, 1);

    let boundary = chain.boundary();

    // Should have three 1-simplices (edges)
    assert_eq!(boundary.items.len(), 3);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.items.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_boundary_squared_is_zero() {
    // Verify that ∂² = 0 for a triangle
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(triangle.clone());
    let chain = Chain::from_item_and_coeff(&complex, triangle, 1);

    let boundary = chain.boundary();
    let boundary_squared = boundary.boundary();

    // Boundary of boundary should be empty (∂² = 0)
    assert_eq!(boundary_squared.items.len(), 0);
    assert_eq!(boundary_squared.coefficients.len(), 0);
  }

  #[test]
  fn test_complex_chain_operations() {
    // Create a 2-chain with two triangles sharing an edge
    let triangle1 = Simplex::new(2, vec![0, 1, 2]);
    let triangle2 = Simplex::new(2, vec![1, 2, 3]);
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(triangle1.clone());
    complex.join_simplex(triangle2.clone());

    let chain1 = Chain::from_item_and_coeff(&complex, triangle1, 1);
    let chain2 = Chain::from_item_and_coeff(&complex, triangle2, -1);

    let combined_chain = chain1 + chain2;
    let boundary = combined_chain.boundary();

    // The boundary should have 4 edges (the shared edge [1,2] cancels out)
    assert_eq!(boundary.items.len(), 4);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.items.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 3]));
    assert!(edge_vertices.contains(&vec![2, 3]));

    // The shared edge [1,2] should not be present because its coefficients cancel
    assert!(!edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_simplices_by_dimension_basic() {
    let mut complex = SimplicialComplex::new();

    // Adding a 2-simplex will also add its 1-simplex faces and 0-simplex vertices
    // due to the behavior of join_simplex.
    let s2_v012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_simplex(s2_v012.clone());

    // Expected 0-simplices (vertices)
    let s0_v0 = Simplex::new(0, vec![0]);
    let s0_v1 = Simplex::new(0, vec![1]);
    let s0_v2 = Simplex::new(0, vec![2]);

    // Expected 1-simplices (edges)
    let s1_v01 = Simplex::new(1, vec![0, 1]); // face of s2_v012
    let s1_v02 = Simplex::new(1, vec![0, 2]); // face of s2_v012
    let s1_v12 = Simplex::new(1, vec![1, 2]); // face of s2_v012

    // Check dimension 2
    let dim2_simplices = complex.simplices_by_dimension(2).expect("Dim 2 should exist");
    assert_eq!(dim2_simplices.len(), 1, "Should be one 2-simplex");
    assert!(dim2_simplices.contains(&s2_v012), "Missing 2-simplex [0,1,2]");

    // Check dimension 1
    let dim1_simplices = complex.simplices_by_dimension(1).expect("Dim 1 should exist");
    assert_eq!(dim1_simplices.len(), 3, "Should be three 1-simplices");
    assert!(dim1_simplices.contains(&s1_v01), "Missing 1-simplex [0,1]");
    assert!(dim1_simplices.contains(&s1_v02), "Missing 1-simplex [0,2]");
    assert!(dim1_simplices.contains(&s1_v12), "Missing 1-simplex [1,2]");

    // Check dimension 0
    let dim0_simplices = complex.simplices_by_dimension(0).expect("Dim 0 should exist");
    assert_eq!(dim0_simplices.len(), 3, "Should be three 0-simplices");
    assert!(dim0_simplices.contains(&s0_v0), "Missing 0-simplex [0]");
    assert!(dim0_simplices.contains(&s0_v1), "Missing 0-simplex [1]");
    assert!(dim0_simplices.contains(&s0_v2), "Missing 0-simplex [2]");
  }

  fn test_homology_point_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let p0 = Simplex::new(0, vec![0]);
    complex.join_simplex(p0);

    // H_0
    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");

    // H_1
    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Should have no generators for field {:?}",
      std::any::type_name::<F>()
    );

    // H_2
    let h2 = complex.homology::<F>(2);
    assert_eq!(
      h2.betti_number,
      0,
      "H2: Betti number for a point should be 0 for field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_point_all_fields() {
    test_homology_point_generic::<Boolean>();
    test_homology_point_generic::<Mod7>();
  }

  fn test_homology_edge_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let edge01 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(edge01);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti for an edge");
    assert_eq!(h0.homology_generators.len(), 1, "H0: One generator");

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti for an edge");
    assert!(
      h1.homology_generators.is_empty(),
      "H1: No generators for edge field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_edge_all_fields() {
    test_homology_edge_generic::<Boolean>();
    test_homology_edge_generic::<Mod7>();
  }

  fn test_homology_two_disjoint_points_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let p0_s = Simplex::new(0, vec![0]);
    let p1_s = Simplex::new(0, vec![1]);
    complex.join_simplex(p0_s);
    complex.join_simplex(p1_s);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 2, "H0: Betti for two points");
    assert_eq!(h0.homology_generators.len(), 2, "H0: Two generators");

    let h1 = complex.homology::<F>(1);
    assert_eq!(
      h1.betti_number,
      0,
      "H1: Betti for two points field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_two_disjoint_points_all_fields() {
    test_homology_two_disjoint_points_generic::<Boolean>();
    test_homology_two_disjoint_points_generic::<Mod7>();
  }

  fn test_homology_filled_triangle_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let triangle012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_simplex(triangle012);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for triangle field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(
      h1.betti_number,
      0,
      "H1: Betti for filled triangle field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      h1.homology_generators.is_empty(),
      "H1: No 1D generators field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(
      h2.betti_number,
      0,
      "H2: Betti for filled triangle field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      h2.homology_generators.is_empty(),
      "H2: No 2D generators field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_filled_triangle_all_fields() {
    test_homology_filled_triangle_generic::<Boolean>();
    test_homology_filled_triangle_generic::<Mod7>();
  }

  fn test_homology_circle_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let s01 = Simplex::new(1, vec![0, 1]);
    let s12 = Simplex::new(1, vec![1, 2]);
    let s02 = Simplex::new(1, vec![0, 2]);

    complex.join_simplex(s01);
    complex.join_simplex(s12);
    complex.join_simplex(s02);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for circle field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 1, "H1: Betti for circle field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h1.homology_generators.len(),
      1,
      "H1: One 1D generator field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for circle field {:?}", std::any::type_name::<F>());
  }

  #[test]
  fn test_homology_circle_all_fields() {
    test_homology_circle_generic::<Boolean>();
    test_homology_circle_generic::<Mod7>();
  }

  fn test_homology_sphere_surface_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();

    let f1 = Simplex::new(2, vec![0, 1, 2]);
    let f2 = Simplex::new(2, vec![0, 1, 3]);
    let f3 = Simplex::new(2, vec![0, 2, 3]);
    let f4 = Simplex::new(2, vec![1, 2, 3]);

    complex.join_simplex(f1);
    complex.join_simplex(f2);
    complex.join_simplex(f3);
    complex.join_simplex(f4);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti sphere field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti sphere field {:?}", std::any::type_name::<F>());
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Generators sphere field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 1, "H2: Betti sphere field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h2.homology_generators.len(),
      1,
      "H2: One 2D generator field {:?}",
      std::any::type_name::<F>()
    );

    let h3 = complex.homology::<F>(3);
    assert_eq!(h3.betti_number, 0, "H3: Betti sphere field {:?}", std::any::type_name::<F>());
  }

  #[test]
  fn test_homology_sphere_surface_all_fields() {
    test_homology_sphere_surface_generic::<Boolean>();
    test_homology_sphere_surface_generic::<Mod7>();
  }

  #[ignore = "TODO: Implement neighborhood for simplicial complex"]
  #[test]
  fn test_simplex_neighborhood() {
    let mut complex = SimplicialComplex::new();
    let s0 = Simplex::new(0, vec![0]);
    let s1 = Simplex::new(0, vec![1]);
    let s01 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(s0.clone());
    complex.join_simplex(s1.clone());
    complex.join_simplex(s01.clone());

    let neighborhood = complex.neighborhood(&s0);

    dbg!(&neighborhood);
    assert!(neighborhood.contains(&s01));
    assert!(neighborhood.len() == 1);

    let neighborhood = complex.neighborhood(&s1);
    assert!(neighborhood.contains(&s01));
    assert!(neighborhood.len() == 1);

    let neighborhood = complex.neighborhood(&s01);
    assert!(neighborhood.is_empty());
  }
}
