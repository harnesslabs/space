// TODO: Verify the homology generators are correct.

use std::fmt::Debug;

use harness_algebra::{arithmetic::Boolean, modular, prime_field, rings::Field};

use super::*; // Make sure Mod7 is in scope here

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
  // Create two chains with different simplices
  let simplex1 = Simplex::new(1, vec![0, 1]);
  let simplex2 = Simplex::new(1, vec![1, 2]);

  let chain1 = Chain::from_simplex_and_coeff(simplex1, 1);
  let chain2 = Chain::from_simplex_and_coeff(simplex2, 2);

  let result = chain1 + chain2;

  assert_eq!(result.simplices.len(), 2);
  assert_eq!(result.coefficients.len(), 2);
  assert_eq!(result.simplices[0].vertices(), &[0, 1]);
  assert_eq!(result.simplices[1].vertices(), &[1, 2]);
  assert_eq!(result.coefficients[0], 1);
  assert_eq!(result.coefficients[1], 2);
}

#[test]
fn test_chain_addition_same_simplex() {
  // Create two chains with the same simplex
  let simplex1 = Simplex::new(1, vec![0, 1]);
  let simplex2 = Simplex::new(1, vec![0, 1]);

  let chain1 = Chain::from_simplex_and_coeff(simplex1, 2);
  let chain2 = Chain::from_simplex_and_coeff(simplex2, 3);

  let result = chain1 + chain2;

  assert_eq!(result.simplices.len(), 1);
  assert_eq!(result.coefficients.len(), 1);
  assert_eq!(result.simplices[0].vertices(), &[0, 1]);
  assert_eq!(result.coefficients[0], 5); // 2 + 3 = 5
}

#[test]
fn test_chain_addition_canceling_coefficients() {
  // Create two chains with the same simplex but opposite coefficients
  let simplex1 = Simplex::new(1, vec![0, 1]);
  let simplex2 = Simplex::new(1, vec![0, 1]);

  let chain1 = Chain::from_simplex_and_coeff(simplex1, 2);
  let chain2 = Chain::from_simplex_and_coeff(simplex2, -2);

  let result = chain1 + chain2;

  // The result should be empty since the coefficients cancel out
  assert_eq!(result.simplices.len(), 0);
  assert_eq!(result.coefficients.len(), 0);
}

#[test]
fn test_chain_boundary_edge() {
  // The boundary of an edge is its two vertices with opposite signs
  let edge = Simplex::new(1, vec![0, 1]);
  let chain = Chain::from_simplex_and_coeff(edge, 1);

  let boundary = chain.boundary();

  // Should have two 0-simplices (vertices)
  assert_eq!(boundary.simplices.len(), 2);
  assert_eq!(boundary.coefficients.len(), 2);

  // Verify the vertices
  assert!(boundary.simplices.iter().any(|s| s.vertices().contains(&0)));
  assert!(boundary.simplices.iter().any(|s| s.vertices().contains(&1)));
  assert!(boundary.simplices.len() == 2);

  // Verify opposite signs (exact sign depends on your implementation)
  assert_eq!(boundary.coefficients[0] + boundary.coefficients[1], 0);
}

#[test]
fn test_chain_boundary_triangle() {
  // The boundary of a triangle is its three edges
  let triangle = Simplex::new(2, vec![0, 1, 2]);
  let chain = Chain::from_simplex_and_coeff(triangle, 1);

  let boundary = chain.boundary();

  // Should have three 1-simplices (edges)
  assert_eq!(boundary.simplices.len(), 3);

  // Verify the edges
  let edge_vertices: Vec<Vec<usize>> =
    boundary.simplices.iter().map(|s| s.vertices().to_vec()).collect();

  assert!(edge_vertices.contains(&vec![0, 1]));
  assert!(edge_vertices.contains(&vec![0, 2]));
  assert!(edge_vertices.contains(&vec![1, 2]));
}

#[test]
fn test_boundary_squared_is_zero() {
  // Verify that ∂² = 0 for a triangle
  let triangle = Simplex::new(2, vec![0, 1, 2]);
  let chain = Chain::from_simplex_and_coeff(triangle, 1);

  let boundary = chain.boundary();
  let boundary_squared = boundary.boundary();

  // Boundary of boundary should be empty (∂² = 0)
  assert_eq!(boundary_squared.simplices.len(), 0);
  assert_eq!(boundary_squared.coefficients.len(), 0);
}

#[test]
fn test_complex_chain_operations() {
  // Create a 2-chain with two triangles sharing an edge
  let triangle1 = Simplex::new(2, vec![0, 1, 2]);
  let triangle2 = Simplex::new(2, vec![1, 2, 3]);

  let chain1 = Chain::from_simplex_and_coeff(triangle1, 1);
  let chain2 = Chain::from_simplex_and_coeff(triangle2, -1);

  let combined_chain = chain1 + chain2;
  let boundary = combined_chain.boundary();

  // The boundary should have 4 edges (the shared edge [1,2] cancels out)
  assert_eq!(boundary.simplices.len(), 4);

  // Verify the edges
  let edge_vertices: Vec<Vec<usize>> =
    boundary.simplices.iter().map(|s| s.vertices().to_vec()).collect();

  assert!(edge_vertices.contains(&vec![0, 1]));
  assert!(edge_vertices.contains(&vec![0, 2]));
  assert!(edge_vertices.contains(&vec![1, 3]));
  assert!(edge_vertices.contains(&vec![2, 3]));

  // The shared edge [1,2] should not be present because its coefficients cancel
  assert!(!edge_vertices.contains(&vec![1, 2]));
}

#[test]
fn test_simplicial_complex_boundary() {
  let mut complex = SimplicialComplex::new();
  complex.join_simplex(Simplex::new(2, vec![0, 1, 2]));
  complex.join_simplex(Simplex::new(2, vec![1, 2, 3]));
  let boundary: Chain<i32> = complex.boundary(2);
  assert_eq!(boundary.simplices.len(), 4);
  assert_eq!(boundary.coefficients.len(), 4);
  assert_eq!(boundary.simplices[0].vertices(), &[0, 2]);
  assert_eq!(boundary.simplices[1].vertices(), &[0, 1]);
  assert_eq!(boundary.simplices[2].vertices(), &[2, 3]);
  assert_eq!(boundary.simplices[3].vertices(), &[1, 3]);
  assert_eq!(boundary.coefficients[0], -1);
  assert_eq!(boundary.coefficients[1], 1);
  assert_eq!(boundary.coefficients[2], -1);
  assert_eq!(boundary.coefficients[3], 1);
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
  complex.join_simplex(p0.clone());

  // H_0
  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.dimension, 0, "H0: Dimension check");
  assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
  assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");
  let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, F::one());
  assert!(
    h0.homology_generators.contains(&expected_gen_h0),
    "H0: Generator mismatch for field {:?}",
    std::any::type_name::<F>()
  );

  // H_1
  let h1 = complex.compute_homology::<F>(1);
  assert_eq!(h1.dimension, 1, "H1: Dimension check");
  assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
  assert!(
    h1.homology_generators.is_empty(),
    "H1: Should have no generators for field {:?}",
    std::any::type_name::<F>()
  );

  // H_2
  let h2 = complex.compute_homology::<F>(2);
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
  complex.join_simplex(edge01.clone());

  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.dimension, 0, "H0: Dimension check");
  assert_eq!(h0.betti_number, 1, "H0: Betti for an edge");
  assert_eq!(h0.homology_generators.len(), 1, "H0: One generator");
  let p0 = Simplex::new(0, vec![0]);
  let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, F::one());
  assert!(
    h0.homology_generators.contains(&expected_gen_h0),
    "H0: Generator for edge field {:?}",
    std::any::type_name::<F>()
  );

  let h1 = complex.compute_homology::<F>(1);
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
  complex.join_simplex(p0_s.clone());
  complex.join_simplex(p1_s.clone());

  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.dimension, 0, "H0: Dimension check");
  assert_eq!(h0.betti_number, 2, "H0: Betti for two points");
  assert_eq!(h0.homology_generators.len(), 2, "H0: Two generators");
  let expected_gen1_h0 = Chain::from_simplex_and_coeff(p0_s, F::one());
  let expected_gen2_h0 = Chain::from_simplex_and_coeff(p1_s, F::one());
  assert!(
    h0.homology_generators.contains(&expected_gen1_h0),
    "H0: Gen [0] missing field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    h0.homology_generators.contains(&expected_gen2_h0),
    "H0: Gen [1] missing field {:?}",
    std::any::type_name::<F>()
  );

  let h1 = complex.compute_homology::<F>(1);
  assert_eq!(h1.betti_number, 0, "H1: Betti for two points field {:?}", std::any::type_name::<F>());
}

#[test]
fn test_homology_two_disjoint_points_all_fields() {
  test_homology_two_disjoint_points_generic::<Boolean>();
  test_homology_two_disjoint_points_generic::<Mod7>();
}

fn test_homology_filled_triangle_generic<F: TestField>() {
  let mut complex = SimplicialComplex::new();
  let triangle012 = Simplex::new(2, vec![0, 1, 2]);
  complex.join_simplex(triangle012.clone());

  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.betti_number, 1, "H0: Betti for triangle field {:?}", std::any::type_name::<F>());

  let h1 = complex.compute_homology::<F>(1);
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

  let h2 = complex.compute_homology::<F>(2);
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

  complex.join_simplex(s01.clone());
  complex.join_simplex(s12.clone());
  complex.join_simplex(s02.clone());

  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.betti_number, 1, "H0: Betti for circle field {:?}", std::any::type_name::<F>());

  let h1 = complex.compute_homology::<F>(1);
  assert_eq!(h1.betti_number, 1, "H1: Betti for circle field {:?}", std::any::type_name::<F>());
  assert_eq!(
    h1.homology_generators.len(),
    1,
    "H1: One 1D generator field {:?}",
    std::any::type_name::<F>()
  );

  let generator_h1 = h1.homology_generators[0].clone();
  assert_eq!(
    generator_h1.simplices.len(),
    3,
    "H1: Circle generator 3 simplices field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h1.simplices.contains(&s01),
    "H1: Gen missing s01 field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h1.simplices.contains(&s12),
    "H1: Gen missing s12 field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h1.simplices.contains(&s02),
    "H1: Gen missing s02 field {:?}",
    std::any::type_name::<F>()
  );

  // Check that all coefficients of the generator are F::one() or -F::one().
  assert!(
    !generator_h1.coefficients.is_empty(),
    "H1: Generator should have coefficients for field {:?}",
    std::any::type_name::<F>()
  );
  let one = F::one();
  let neg_one = -F::one();

  for coeff in generator_h1.coefficients.iter() {
    assert!(
      *coeff == one || *coeff == neg_one,
      "H1: Each coefficient in the generator should be F::one() or -F::one(). Found: {:?} for \
       field {:?}",
      *coeff,
      std::any::type_name::<F>()
    );
  }

  assert!(
    generator_h1.boundary().simplices.is_empty(),
    "H1: Generator boundary zero field {:?}",
    std::any::type_name::<F>()
  );

  let h2 = complex.compute_homology::<F>(2);
  assert_eq!(h2.betti_number, 0, "H2: Betti for circle field {:?}", std::any::type_name::<F>());
}

#[test]
fn test_homology_circle_all_fields() {
  test_homology_circle_generic::<Boolean>();
  test_homology_circle_generic::<Mod7>();
}

fn test_homology_sphere_surface_generic<F: TestField>() {
  let mut complex = SimplicialComplex::new();
  let f012 = Simplex::new(2, vec![0, 1, 2]);
  let f013 = Simplex::new(2, vec![0, 1, 3]);
  let f023 = Simplex::new(2, vec![0, 2, 3]);
  let f123 = Simplex::new(2, vec![1, 2, 3]);

  complex.join_simplex(f012.clone());
  complex.join_simplex(f013.clone());
  complex.join_simplex(f023.clone());
  complex.join_simplex(f123.clone());

  let h0 = complex.compute_homology::<F>(0);
  assert_eq!(h0.betti_number, 1, "H0: Betti sphere field {:?}", std::any::type_name::<F>());

  let h1 = complex.compute_homology::<F>(1);
  assert_eq!(h1.betti_number, 0, "H1: Betti sphere field {:?}", std::any::type_name::<F>());
  assert!(
    h1.homology_generators.is_empty(),
    "H1: Generators sphere field {:?}",
    std::any::type_name::<F>()
  );

  let h2 = complex.compute_homology::<F>(2);
  assert_eq!(h2.betti_number, 1, "H2: Betti sphere field {:?}", std::any::type_name::<F>());
  assert_eq!(
    h2.homology_generators.len(),
    1,
    "H2: One 2D generator field {:?}",
    std::any::type_name::<F>()
  );

  let generator_h2 = h2.homology_generators[0].clone();
  assert_eq!(
    generator_h2.simplices.len(),
    4,
    "H2: Sphere generator 4 faces field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h2.simplices.contains(&f012),
    "H2: Gen missing f012 field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h2.simplices.contains(&f013),
    "H2: Gen missing f013 field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h2.simplices.contains(&f023),
    "H2: Gen missing f023 field {:?}",
    std::any::type_name::<F>()
  );
  assert!(
    generator_h2.simplices.contains(&f123),
    "H2: Gen missing f123 field {:?}",
    std::any::type_name::<F>()
  );

  // Check that all coefficients of the generator are F::one() or -F::one().
  assert!(
    !generator_h2.coefficients.is_empty(),
    "H2: Generator should have coefficients for field {:?}",
    std::any::type_name::<F>()
  );
  let one = F::one();
  let neg_one = -F::one();

  for coeff in generator_h2.coefficients.iter() {
    assert!(
      *coeff == one || *coeff == neg_one,
      "H2: Each coefficient in the generator should be F::one() or -F::one(). Found: {:?} for \
       field {:?}",
      *coeff,
      std::any::type_name::<F>()
    );
  }

  assert!(
    generator_h2.boundary().simplices.is_empty(),
    "H2: Generator boundary zero field {:?}",
    std::any::type_name::<F>()
  );

  let h3 = complex.compute_homology::<F>(3);
  assert_eq!(h3.betti_number, 0, "H3: Betti sphere field {:?}", std::any::type_name::<F>());
}

#[test]
fn test_homology_sphere_surface_all_fields() {
  test_homology_sphere_surface_generic::<Boolean>();
  test_homology_sphere_surface_generic::<Mod7>();
}
