//! A module for working with simplicial complexes and computing their boundaries.
//!
//! This module provides data structures and algorithms for working with simplicial complexes,
//! including operations for computing boundaries and manipulating chains of simplices.
//! The implementation ensures that vertices in simplices are always stored in sorted order.

use std::ops::{Add, Neg};

use itertools::Itertools;
use num::{One, Zero};

/// A simplex represents a k-dimensional geometric object that is the convex hull of k+1 vertices.
///
/// For example:
/// - A 0-simplex is a point
/// - A 1-simplex is a line segment
/// - A 2-simplex is a triangle
/// - A 3-simplex is a tetrahedron
///
/// # Fields
/// * `vertices` - A sorted vector of vertex indices that define the simplex
/// * `dimension` - The dimension of the simplex (number of vertices - 1)
#[derive(Clone, Debug)]
pub struct Simplex {
  vertices: Vec<usize>,
  dimension: usize,
}

impl PartialEq for Simplex {
  fn eq(&self, other: &Self) -> bool {
    self.vertices == other.vertices
  }
}

impl Simplex {
  /// Creates a new simplex from k+1 vertices where k is the dimension.
  ///
  /// # Arguments
  /// * `dimension` - The dimension of the simplex
  /// * `vertices` - A vector of vertex indices
  ///
  /// # Panics
  /// * If the number of vertices does not equal dimension + 1
  /// * If any vertex indices are repeated
  pub fn new(dimension: usize, vertices: Vec<usize>) -> Self {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices: vertices.into_iter().sorted().collect(), dimension }
  }

  /// Returns a reference to the sorted vertices of the simplex.
  pub fn vertices(&self) -> &[usize] {
    &self.vertices
  }

  /// Returns the dimension of the simplex.
  pub fn dimension(&self) -> usize {
    self.dimension
  }

  /// Computes all (dimension-1)-dimensional faces of this simplex.
  ///
  /// For example, a triangle's faces are its three edges.
  pub fn faces(&self) -> Vec<Simplex> {
    self
      .vertices
      .clone()
      .into_iter()
      .combinations(self.dimension)
      .map(|v| Self::new(self.dimension - 1, v))
      .collect()
  }
}

/// A simplicial complex represents a collection of simplices that are properly glued together.
///
/// The complex stores simplices grouped by dimension, where each dimension's simplices
/// are stored in a vector at the corresponding index.
#[derive(Debug, Default)]
pub struct SimplicialComplex {
  /// Vector of simplices grouped by dimension
  simplices: Vec<Vec<Simplex>>,
}

impl SimplicialComplex {
  /// Creates a new empty simplicial complex.
  pub fn new() -> Self {
    Self { simplices: vec![] }
  }

  /// Adds a simplex and all its faces to the complex.
  ///
  /// If the simplex is already present, it will not be added again.
  /// This method recursively adds all faces of the simplex as well.
  pub fn join_simplex(&mut self, simplex: Simplex) {
    while self.simplices.len() <= simplex.dimension() {
      self.simplices.push(Vec::new());
    }
    if self.simplices[simplex.dimension()].contains(&simplex) {
      return;
    }

    if simplex.dimension() > 0 {
      for face in simplex.faces() {
        self.join_simplex(face);
      }
    }
    self.simplices[simplex.dimension()].push(simplex);
  }

  /// Computes the boundary of all simplices of a given dimension in the complex.
  ///
  /// # Type Parameters
  /// * `R` - The coefficient ring type (must implement Clone, Neg, Add, Zero, and One)
  ///
  /// # Arguments
  /// * `dimension` - The dimension of simplices whose boundary to compute
  ///
  /// # Returns
  /// A chain representing the boundary. If dimension is 0 or exceeds the maximum
  /// dimension in the complex, returns an empty chain.
  ///
  /// # Note
  /// This implementation assumes at most two simplices share any given face.
  pub fn boundary<R: Clone + Neg<Output = R> + Add<Output = R> + Zero + One>(
    &self,
    dimension: usize,
  ) -> Chain<R> {
    if dimension == 0 || dimension >= self.simplices.len() {
      return Chain::new();
    }
    let mut chain = Chain::new();
    let simplices = self.simplices[dimension].clone();
    for simplex in simplices {
      if chain.simplices.iter().flat_map(Simplex::faces).any(|f| simplex.faces().contains(&f)) {
        chain = chain + Chain::from_simplex_and_coeff(simplex, -R::one());
      } else {
        chain = chain + Chain::from_simplex_and_coeff(simplex, R::one());
      }
    }
    chain.boundary()
  }
}

/// A chain represents a formal sum of simplices with coefficients from a ring.
///
/// # Type Parameters
/// * `R` - The coefficient ring type
#[derive(Clone, Debug, Default)]
pub struct Chain<R> {
  /// The simplices in the chain
  simplices: Vec<Simplex>,
  /// The coefficients corresponding to each simplex
  coefficients: Vec<R>,
}

impl<R> Chain<R> {
  /// Creates a new empty chain.
  pub fn new() -> Self {
    Self { simplices: vec![], coefficients: vec![] }
  }

  /// Creates a new chain with a single simplex and coefficient.
  pub fn from_simplex_and_coeff(simplex: Simplex, coeff: R) -> Self {
    Self { simplices: vec![simplex], coefficients: vec![coeff] }
  }

  /// Computes the boundary of this chain.
  ///
  /// The boundary operator satisfies the property that ∂² = 0,
  /// meaning the boundary of a boundary is empty.
  pub fn boundary(&self) -> Self
  where
    R: Clone + Neg<Output = R> + Add<Output = R> + Zero,
  {
    let mut boundary = Self::new();
    for (coeff, simplex) in self.coefficients.clone().into_iter().zip(self.simplices.iter()) {
      for i in 0..=simplex.dimension() {
        let mut vertices = simplex.vertices().to_vec();
        let _ = vertices.remove(i);
        let face = Simplex::new(simplex.dimension() - 1, vertices);
        let chain = Self::from_simplex_and_coeff(
          face,
          if i % 2 == 0 { coeff.clone() } else { -coeff.clone() },
        );
        boundary = boundary + chain;
      }
    }
    boundary
  }
}

impl<R: Clone + PartialEq> PartialEq for Chain<R> {
  /// Checks if two chains are equal.
  ///
  /// Two chains are equal if they have the same simplices with the same coefficients,
  /// taking into account the orientation of the simplices.
  fn eq(&self, other: &Self) -> bool {
    let self_chain = self.coefficients.clone().into_iter().zip(self.simplices.iter());
    let other_chain = other.coefficients.clone().into_iter().zip(other.simplices.iter());

    self_chain.zip(other_chain).all(|((coeff_a, simplex_a), (coeff_b, simplex_b))| {
      coeff_a == coeff_b
        && permutation_sign(simplex_a.vertices()) == permutation_sign(simplex_b.vertices())
        && simplex_a.vertices() == simplex_b.vertices()
    })
  }
}

impl<R: Add<Output = R> + Neg<Output = R> + Clone + Zero> Add for Chain<R> {
  type Output = Self;

  /// Adds two chains together.
  ///
  /// The addition combines like terms (simplices) by adding their coefficients.
  /// Terms with zero coefficients are removed from the result.
  ///
  /// # Note
  /// This implementation assumes the chains contain simplices of the same dimension.
  fn add(self, other: Self) -> Self::Output {
    let mut result_simplices = Vec::new();
    let mut result_coefficients: Vec<R> = Vec::new();

    // Add all simplices from self
    for (idx, simplex) in self.simplices.iter().enumerate() {
      let coefficient = self.coefficients[idx].clone();

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, res_simplex) in result_simplices.iter().enumerate() {
        if simplex == res_simplex {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx].clone() + coefficient.clone();
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_simplices.push(simplex.clone());
        result_coefficients.push(coefficient);
      }
    }

    // Add all simplices from other
    for (idx, simplex) in other.simplices.iter().enumerate() {
      let coefficient = other.coefficients[idx].clone();

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, res_simplex) in result_simplices.iter().enumerate() {
        if simplex == res_simplex {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx].clone() + coefficient.clone();
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_simplices.push(simplex.clone());
        result_coefficients.push(coefficient);
      }
    }

    // Filter out zero coefficients
    let mut i = 0;
    while i < result_coefficients.len() {
      if result_coefficients[i].is_zero() {
        // Remove zero coefficients and their corresponding simplices
        result_coefficients.remove(i);
        result_simplices.remove(i);
        // Don't increment i since we've shifted the array
      } else {
        i += 1;
      }
    }

    Self { simplices: result_simplices, coefficients: result_coefficients }
  }
}

/// Represents whether a permutation is odd or even.
#[derive(Debug, PartialEq, Eq)]
pub enum Permutation {
  /// An odd permutation
  Odd,
  /// An even permutation
  Even,
}

/// Computes the sign of a permutation by counting inversions.
///
/// # Arguments
/// * `item` - A slice of ordered items
///
/// # Returns
/// `Permutation::Even` if the number of inversions is even,
/// `Permutation::Odd` if the number of inversions is odd.
pub fn permutation_sign<V: Ord>(item: &[V]) -> Permutation {
  let mut count = 0;
  for i in 0..item.len() {
    for j in i + 1..item.len() {
      if item[i] > item[j] {
        count += 1;
      }
    }
  }
  if count % 2 == 0 {
    Permutation::Even
  } else {
    Permutation::Odd
  }
}

#[cfg(test)]
mod tests {
  use super::*;

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
    assert_eq!(complex.simplices[2].len(), 1);
    assert_eq!(complex.simplices[1].len(), 3);
    assert_eq!(complex.simplices[0].len(), 3);
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
}
