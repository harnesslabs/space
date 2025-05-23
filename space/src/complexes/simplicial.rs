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

  /// Computes the faces with their correct orientation coefficients for the simplicial boundary
  /// operator.
  ///
  /// For a k-simplex σ = [v_0, v_1, ..., v_k], the boundary is:
  /// ∂σ = Σ_{i=0}^k (-1)^i [v_0, ..., v̂_i, ..., v_k]
  /// where v̂_i means vertex v_i is omitted.
  fn boundary_with_orientations(&self) -> Vec<(Self, i32)> {
    if self.dimension == 0 {
      return Vec::new();
    }

    let mut faces_with_orientations = Vec::new();

    // For each vertex position, create a face by omitting that vertex
    for (i, _) in self.vertices.iter().enumerate() {
      let mut face_vertices = self.vertices.clone();
      face_vertices.remove(i);
      let face = Self::from_vertices(face_vertices);

      // Alternating sign: (-1)^i
      let orientation = if i % 2 == 0 { 1 } else { -1 };
      faces_with_orientations.push((face, orientation));
    }

    faces_with_orientations
  }

  fn id(&self) -> Option<usize> { self.id }

  fn same_content(&self, other: &Self) -> bool { self.same_content(other) }

  fn with_id(&self, new_id: usize) -> Self { self.clone().with_id(new_id) }
}

#[cfg(test)]
mod tests {
  use std::fmt::Debug;

  use harness_algebra::{algebras::boolean::Boolean, modular, prime_field, rings::Field};

  use super::*;
  use crate::complexes::Complex;

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);
  // Helper trait bound alias for tests
  trait TestField: Field + Copy + Debug {}
  impl<T: Field + Copy + Debug> TestField for T {}

  #[test]
  fn test_simplex_construction() {
    let vertex = Simplex::new(0, vec![0]);
    assert_eq!(vertex.dimension(), 0);
    assert_eq!(vertex.vertices(), &[0]);
    assert_eq!(vertex.id(), None);

    let edge = Simplex::new(1, vec![1, 0]); // Will be sorted
    assert_eq!(edge.dimension(), 1);
    assert_eq!(edge.vertices(), &[0, 1]); // Should be sorted
    assert_eq!(edge.id(), None);

    let triangle = Simplex::from_vertices(vec![2, 0, 1]);
    assert_eq!(triangle.dimension(), 2);
    assert_eq!(triangle.vertices(), &[0, 1, 2]); // Should be sorted
  }

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
  fn test_simplex_with_id() {
    let simplex = Simplex::new(1, vec![0, 1]);
    assert_eq!(simplex.id(), None);

    let simplex_with_id = simplex.with_id(42);
    assert_eq!(simplex_with_id.id(), Some(42));
    assert_eq!(simplex_with_id.vertices(), &[0, 1]); // Content unchanged
  }

  #[test]
  fn test_simplex_same_content() {
    let s1 = Simplex::new(1, vec![0, 1]);
    let s2 = Simplex::new(1, vec![0, 1]);
    let s3 = Simplex::new(1, vec![0, 2]);
    let s4 = s1.clone().with_id(42); // Same content, different ID

    assert!(s1.same_content(&s2));
    assert!(!s1.same_content(&s3));
    assert!(s1.same_content(&s4)); // Content equality ignores ID
  }

  #[test]
  fn test_simplex_ordering() {
    let s1 = Simplex::new(0, vec![0]);
    let s2 = Simplex::new(0, vec![1]);
    let s3 = Simplex::new(1, vec![0, 1]);

    assert!(s1 < s2); // Same dimension, different vertices
    assert!(s1 < s3); // Different dimension (vertices order first)
  }

  #[test]
  fn test_simplicial_complex_basic() {
    let mut complex: Complex<Simplex> = Complex::new();
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    complex.join_element(triangle);

    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert_eq!(complex.elements_of_dimension(1).len(), 3);
    assert_eq!(complex.elements_of_dimension(0).len(), 3);
  }

  #[test]
  fn test_chain_operations_with_simplices() {
    let mut complex: Complex<Simplex> = Complex::new();

    // Create two edges
    let edge1 = Simplex::new(1, vec![0, 1]);
    let edge2 = Simplex::new(1, vec![1, 2]);
    let added_edge1 = complex.join_element(edge1);
    let added_edge2 = complex.join_element(edge2);

    let chain1 = Chain::from_item_and_coeff(&complex, added_edge1, 1_i32);
    let chain2 = Chain::from_item_and_coeff(&complex, added_edge2, 2_i32);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.coefficients, vec![1, 2]);
  }

  #[test]
  fn test_chain_boundary_operations() {
    let mut complex: Complex<Simplex> = Complex::new();

    // Test edge boundary
    let edge = Simplex::new(1, vec![0, 1]);
    let added_edge = complex.join_element(edge);
    let chain = Chain::from_item_and_coeff(&complex, added_edge, 1);

    let boundary = chain.boundary();
    assert_eq!(boundary.items.len(), 2); // Two vertices
    assert_eq!(boundary.coefficients[0] + boundary.coefficients[1], 0); // Opposite signs

    // Test triangle boundary
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);
    let triangle_chain = Chain::from_item_and_coeff(&complex, added_triangle, 1);

    let triangle_boundary = triangle_chain.boundary();
    assert_eq!(triangle_boundary.items.len(), 3); // Three edges
  }

  #[test]
  fn test_boundary_squared_is_zero() {
    let mut complex: Complex<Simplex> = Complex::new();

    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);
    let chain = Chain::from_item_and_coeff(&complex, added_triangle, 1);

    let boundary = chain.boundary();
    let boundary_squared = boundary.boundary();

    // Boundary of boundary should be empty (∂² = 0)
    assert_eq!(boundary_squared.items.len(), 0);
    assert_eq!(boundary_squared.coefficients.len(), 0);
  }

  #[test]
  fn test_complex_chain_operations() {
    let mut complex: Complex<Simplex> = Complex::new();

    // Create two triangles sharing an edge
    let triangle1 = Simplex::new(2, vec![0, 1, 2]);
    let triangle2 = Simplex::new(2, vec![1, 2, 3]);
    let added_t1 = complex.join_element(triangle1);
    let added_t2 = complex.join_element(triangle2);

    let chain1 = Chain::from_item_and_coeff(&complex, added_t1, 1);
    let chain2 = Chain::from_item_and_coeff(&complex, added_t2, -1);

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

  fn test_simplicial_homology_point_generic<F: TestField>() {
    let mut complex: Complex<Simplex> = Complex::new();
    let p0 = Simplex::new(0, vec![0]);
    complex.join_element(p0);

    // H_0
    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");

    // H_1
    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
    assert!(h1.homology_generators.is_empty(), "H1: Should have no generators");
  }

  #[test]
  fn test_simplicial_homology_point_all_fields() {
    test_simplicial_homology_point_generic::<Boolean>();
    test_simplicial_homology_point_generic::<Mod7>();
  }

  fn test_simplicial_homology_circle_generic<F: TestField>() {
    let mut complex: Complex<Simplex> = Complex::new();
    let s01 = Simplex::new(1, vec![0, 1]);
    let s12 = Simplex::new(1, vec![1, 2]);
    let s02 = Simplex::new(1, vec![0, 2]);

    complex.join_element(s01);
    complex.join_element(s12);
    complex.join_element(s02);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for circle");

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 1, "H1: Betti for circle");
    assert_eq!(h1.homology_generators.len(), 1, "H1: One 1D generator");

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for circle");
  }

  #[test]
  fn test_simplicial_homology_circle_all_fields() {
    test_simplicial_homology_circle_generic::<Boolean>();
    test_simplicial_homology_circle_generic::<Mod7>();
  }

  fn test_simplicial_homology_filled_triangle_generic<F: TestField>() {
    let mut complex: Complex<Simplex> = Complex::new();
    let triangle012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_element(triangle012);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for triangle");

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti for filled triangle");
    assert!(h1.homology_generators.is_empty(), "H1: No 1D generators");

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for filled triangle");
    assert!(h2.homology_generators.is_empty(), "H2: No 2D generators");
  }

  #[test]
  fn test_simplicial_homology_filled_triangle_all_fields() {
    test_simplicial_homology_filled_triangle_generic::<Boolean>();
    test_simplicial_homology_filled_triangle_generic::<Mod7>();
  }

  fn test_simplicial_homology_sphere_surface_generic<F: TestField>() {
    let mut complex: Complex<Simplex> = Complex::new();

    let f1 = Simplex::new(2, vec![0, 1, 2]);
    let f2 = Simplex::new(2, vec![0, 1, 3]);
    let f3 = Simplex::new(2, vec![0, 2, 3]);
    let f4 = Simplex::new(2, vec![1, 2, 3]);

    complex.join_element(f1);
    complex.join_element(f2);
    complex.join_element(f3);
    complex.join_element(f4);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti sphere");

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti sphere");
    assert!(h1.homology_generators.is_empty(), "H1: Generators sphere");

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 1, "H2: Betti sphere");
    assert_eq!(h2.homology_generators.len(), 1, "H2: One 2D generator");

    let h3 = complex.homology::<F>(3);
    assert_eq!(h3.betti_number, 0, "H3: Betti sphere");
  }

  #[test]
  fn test_simplicial_homology_sphere_surface_all_fields() {
    test_simplicial_homology_sphere_surface_generic::<Boolean>();
    test_simplicial_homology_sphere_surface_generic::<Mod7>();
  }

  #[test]
  fn test_simplicial_incidence_poset_condition_1() {
    // Condition 1: If [σ : τ] ≠ 0, then σ ⊲ τ and there are no cells between σ and τ
    let mut complex: Complex<Simplex> = Complex::new();

    // Create a triangle
    let v0 = Simplex::new(0, vec![0]);
    let v1 = Simplex::new(0, vec![1]);
    let v2 = Simplex::new(0, vec![2]);
    let e01 = Simplex::new(1, vec![0, 1]);
    let e12 = Simplex::new(1, vec![1, 2]);
    let e02 = Simplex::new(1, vec![0, 2]);
    let triangle = Simplex::new(2, vec![0, 1, 2]);

    let added_triangle = complex.join_element(triangle);

    // Get all elements
    let vertices = complex.elements_of_dimension(0);
    let edges = complex.elements_of_dimension(1);
    let triangles = complex.elements_of_dimension(2);

    // Test that triangle's boundary consists only of direct faces (edges)
    let boundary_with_orientations = added_triangle.boundary_with_orientations();
    for (face, _orientation) in boundary_with_orientations {
      // Each face should be an edge (1-dimensional)
      assert_eq!(face.dimension(), 1);
      // Each face should be in the complex
      assert!(edges.iter().any(|e| e.same_content(&face)));
      // There should be no elements between the triangle and its faces
      // (already guaranteed by construction since triangle is 2D and faces are 1D)
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

    println!("✓ Simplicial Incidence Condition 1: Non-zero incidence implies direct face relation");
  }

  #[test]
  fn test_simplicial_incidence_poset_condition_2() {
    // Condition 2: For any σ ⊲ τ, Σ_γ∈P_X [σ : γ][γ : τ] = 0 (∂² = 0)
    let mut complex: Complex<Simplex> = Complex::new();

    // Test with a triangle
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    // Create a chain from the triangle
    let triangle_chain = Chain::from_item_and_coeff(&complex, added_triangle, 1);

    // Compute boundary of the triangle (should give edges)
    let boundary_1 = triangle_chain.boundary();

    // Compute boundary of the boundary (should be zero)
    let boundary_2 = boundary_1.boundary();

    // ∂² should be zero
    assert_eq!(boundary_2.items.len(), 0, "∂² should be zero for simplicial complex");
    assert_eq!(boundary_2.coefficients.len(), 0, "∂² should have no coefficients");

    // Also test with a tetrahedron
    let tetrahedron = Simplex::new(3, vec![0, 1, 2, 3]);
    let mut tet_complex: Complex<Simplex> = Complex::new();
    let added_tet = tet_complex.join_element(tetrahedron);

    let tet_chain = Chain::from_item_and_coeff(&tet_complex, added_tet, 1);
    let tet_boundary_1 = tet_chain.boundary();
    let tet_boundary_2 = tet_boundary_1.boundary();

    assert_eq!(tet_boundary_2.items.len(), 0, "∂² should be zero for 3D tetrahedron");
  }

  #[test]
  fn test_simplicial_chain_complex_property() {
    // Additional comprehensive test for the chain complex property
    let mut complex: Complex<Simplex> = Complex::new();

    // Create a more complex structure - multiple triangles sharing edges
    let triangle1 = Simplex::new(2, vec![0, 1, 2]);
    let triangle2 = Simplex::new(2, vec![1, 2, 3]);

    let t1 = complex.join_element(triangle1);
    let t2 = complex.join_element(triangle2);

    // Test ∂² = 0 for individual triangles
    let chain1 = Chain::from_item_and_coeff(&complex, t1, 1);
    let chain2 = Chain::from_item_and_coeff(&complex, t2, 1);

    assert_eq!(chain1.boundary().boundary().items.len(), 0);
    assert_eq!(chain2.boundary().boundary().items.len(), 0);

    // Test ∂² = 0 for linear combination
    let combined_chain = chain1.add(chain2);
    assert_eq!(combined_chain.boundary().boundary().items.len(), 0);

    println!("✓ Simplicial chain complex property verified for complex structures");
  }
}
