//! # Dynamic Tensors Module
//!
//! This module provides implementations of tensors with dynamically determined dimensions,
//! focusing on vectors and matrices.
//!
//! ## Overview
//!
//! The dynamic tensors module includes:
//!
//! - `Vector`: A flexible vector implementation with arbitrary dimension
//! - `Matrix`: A clean, ergonomic matrix implementation with block construction support
//!
//! ## Mathematical Foundation
//!
//! Tensors are mathematical objects that generalize vectors and matrices to higher dimensions.
//! The implementations in this module adhere to the algebraic properties of vector spaces
//! and linear transformations over arbitrary fields.
//!
//! ## Examples
//!
//! ```
//! use cova_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{matrix::Matrix, vector::Vector},
//! };
//!
//! // Create vectors
//! let v1 = Vector::from([1.0, 2.0, 3.0]);
//! let v2 = Vector::from([4.0, 5.0, 6.0]);
//!
//! // Create a matrix from rows
//! let matrix = Matrix::from_rows([v1, v2]);
//!
//! // Or using builder pattern
//! let matrix = Matrix::builder().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).build();
//!
//! // Block matrix construction
//! let block_matrix = Matrix::from_blocks(vec![vec![Some(Matrix::identity(2)), None], vec![
//!   None,
//!   Some(Matrix::identity(3)),
//! ]]);
//!
//! // Perform Gaussian elimination to row echelon form
//! let (rref, output) = matrix.into_row_echelon_form();
//! ```

pub use matrix::Matrix;
pub use vector::Vector;

use super::*;

pub mod matrix;
pub mod vector;

/// Computes a basis for the quotient space V/U.
///
/// Given `subspace_vectors` forming a basis for a subspace U, and
/// `space_vectors` forming a basis for a space V (where U is a subspace of V),
/// this method returns a basis for the quotient space V/U.
/// The input vectors are treated as column vectors.
pub fn compute_quotient_basis<F: Field + Copy>(
  subspace_vectors: &[Vector<F>],
  space_vectors: &[Vector<F>],
) -> Vec<Vector<F>> {
  if space_vectors.is_empty() {
    return Vec::new();
  }

  // Determine the common dimension of all vectors from the first space vector.
  // All vectors (subspace and space) must have this same dimension.
  let expected_num_rows = space_vectors[0].dimension();

  // Verify all subspace vectors match this dimension.
  for (idx, vec) in subspace_vectors.iter().enumerate() {
    assert!(
      (vec.dimension() == expected_num_rows),
      "Subspace vector at index {} has dimension {} but expected {}",
      idx,
      vec.dimension(),
      expected_num_rows
    );
  }
  // Verify all other space vectors match this dimension.
  for (idx, vec) in space_vectors.iter().skip(1).enumerate() {
    assert!(
      (vec.dimension() == expected_num_rows),
      "Space vector at index {} (after first) has dimension {} but expected {}",
      idx + 1, // adjust index because of skip(1)
      vec.dimension(),
      expected_num_rows
    );
  }

  // Create matrix from columns
  let mut all_columns = Vec::new();
  all_columns.extend_from_slice(subspace_vectors);
  all_columns.extend_from_slice(space_vectors);

  let matrix = Matrix::from_cols(all_columns);
  let (_, echelon_output) = matrix.into_row_echelon_form();

  let mut quotient_basis: Vec<Vector<F>> = Vec::new();
  let num_subspace_cols = subspace_vectors.len();

  let pivot_cols_set: std::collections::HashSet<usize> =
    echelon_output.pivots.iter().map(|p| p.col).collect();

  for (i, original_space_vector) in space_vectors.iter().enumerate() {
    let augmented_matrix_col_idx = num_subspace_cols + i;
    if pivot_cols_set.contains(&augmented_matrix_col_idx) {
      quotient_basis.push(original_space_vector.clone());
    }
  }

  quotient_basis
}

#[cfg(test)]
mod tests {
  use super::compute_quotient_basis;
  use crate::{fixtures::Mod7, tensors::dynamic::vector::Vector};

  #[test]
  fn test_quotient_simple_span() {
    // V = span{[1,0,0], [0,1,0]}, U = span{[1,0,0]}
    // V/U should be span{[0,1,0]}
    let u1 = Vector::from(vec![Mod7::new(1), Mod7::new(0), Mod7::new(0)]);
    let v_in_u = Vector::from(vec![Mod7::new(1), Mod7::new(0), Mod7::new(0)]);
    let v_new = Vector::from(vec![Mod7::new(0), Mod7::new(1), Mod7::new(0)]);

    let subspace_vectors = vec![u1];
    // Order of space_vectors: putting v_in_u first
    let space_vectors = vec![v_in_u.clone(), v_new.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);

    assert_eq!(quotient_basis.len(), 1, "Quotient basis should have 1 vector");
    assert!(quotient_basis.contains(&v_new), "Quotient basis should contain the new vector");
    assert!(
      !quotient_basis.contains(&v_in_u),
      "Quotient basis should not contain vector already effectively in subspace"
    );
  }

  #[test]
  fn test_quotient_subspace_equals_space() {
    // V = span{[1,0], [0,1]}, U = span{[1,0], [0,1]}
    // V/U should be empty
    let u1 = Vector::from(vec![Mod7::new(1), Mod7::new(0)]);
    let u2 = Vector::from(vec![Mod7::new(0), Mod7::new(1)]);
    // space_vectors are the same as subspace_vectors
    let space_vectors = vec![u1.clone(), u2.clone()];
    let subspace_vectors = vec![u1.clone(), u2.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert_eq!(
      quotient_basis.len(),
      0,
      "Quotient basis should be empty when subspace equals space"
    );
  }

  #[test]
  fn test_quotient_trivial_subspace() {
    // V = span{[1,0], [0,1]}, U = {} (trivial subspace)
    // V/U should be span{[1,0], [0,1]}
    let v1 = Vector::from(vec![Mod7::new(1), Mod7::new(0)]);
    let v2 = Vector::from(vec![Mod7::new(0), Mod7::new(1)]);

    let subspace_vectors: Vec<Vector<Mod7>> = vec![];
    let space_vectors = vec![v1.clone(), v2.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert_eq!(quotient_basis.len(), 2, "Quotient basis size mismatch for trivial subspace");
    assert!(quotient_basis.contains(&v1), "Quotient basis should contain v1 for trivial subspace");
    assert!(quotient_basis.contains(&v2), "Quotient basis should contain v2 for trivial subspace");
  }

  #[test]
  fn test_quotient_dependent_space_vectors() {
    // V = span{[1,0], [2,0], [0,1]}, U = span{[1,0]}
    // [2,0] is dependent on [1,0].
    // V/U should be span{[0,1]}
    let u1 = Vector::from(vec![Mod7::new(1), Mod7::new(0)]);
    let v_in_u = Vector::from(vec![Mod7::new(1), Mod7::new(0)]); // Effectively in U
    let v_dependent_on_u = Vector::from(vec![Mod7::new(2), Mod7::new(0)]); // 2*u1
    let v_new_independent = Vector::from(vec![Mod7::new(0), Mod7::new(1)]);

    let subspace_vectors = vec![u1.clone()];
    let space_vectors = vec![v_in_u.clone(), v_dependent_on_u.clone(), v_new_independent.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);

    assert_eq!(
      quotient_basis.len(),
      1,
      "Quotient basis should have 1 vector for dependent space vectors case"
    );
    assert!(
      quotient_basis.contains(&v_new_independent),
      "Quotient basis should contain the truly new vector"
    );
    assert!(!quotient_basis.contains(&v_in_u));
    assert!(!quotient_basis.contains(&v_dependent_on_u));
  }

  #[test]
  fn test_quotient_space_vectors_dependent_among_themselves_but_new_to_subspace() {
    // U = span{[1,0,0]}
    // V = span{[1,0,0], [0,1,0], [0,2,0], [0,0,1]}
    // Original space_vectors to select from for quotient: [[0,1,0], [0,2,0], [0,0,1]]
    // Expected quotient basis: a basis for span{[0,1,0], [0,0,1]}, chosen from original space
    // vectors. So, should be [[0,1,0], [0,0,1]] if [0,2,0] is correctly identified as dependent
    // on [0,1,0] in context of quotient.
    let u1 = Vector::from(vec![Mod7::new(1), Mod7::new(0), Mod7::new(0)]);

    let v1_new = Vector::from(vec![Mod7::new(0), Mod7::new(1), Mod7::new(0)]);
    let v2_dependent_on_v1 = Vector::from(vec![Mod7::new(0), Mod7::new(2), Mod7::new(0)]);
    let v3_new = Vector::from(vec![Mod7::new(0), Mod7::new(0), Mod7::new(1)]);

    let subspace_vectors = vec![u1];
    // Order: v1_new, then its dependent v2_dependent_on_v1, then independent v3_new
    let space_vectors = vec![v1_new.clone(), v2_dependent_on_v1.clone(), v3_new.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);

    assert_eq!(
      quotient_basis.len(),
      2,
      "Quotient basis size mismatch for internally dependent space vectors"
    );
    assert!(quotient_basis.contains(&v1_new), "Quotient basis should contain v1_new");
    assert!(quotient_basis.contains(&v3_new), "Quotient basis should contain v3_new");
    assert!(
      !quotient_basis.contains(&v2_dependent_on_v1),
      "Quotient basis should not contain v2_dependent_on_v1"
    );
  }

  #[test]
  fn test_quotient_empty_space_vectors() {
    let u1 = Vector::from(vec![Mod7::new(1), Mod7::new(0)]);
    let subspace_vectors = vec![u1];
    let space_vectors: Vec<Vector<Mod7>> = vec![];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(quotient_basis.is_empty(), "Quotient basis should be empty if space_vectors is empty");
  }

  #[test]
  fn test_quotient_zero_dimensional_vectors() {
    let u1_zero_dim = Vector::<Mod7>::new(vec![]);
    let v1_zero_dim = Vector::<Mod7>::new(vec![]);
    let v2_zero_dim = Vector::<Mod7>::new(vec![]);

    let subspace_vectors = vec![u1_zero_dim];
    let space_vectors = vec![v1_zero_dim, v2_zero_dim];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(
      quotient_basis.is_empty(),
      "Quotient basis for zero-dimensional vectors should be empty"
    );

    // Case: subspace empty, space has 0-dim vectors
    let subspace_vectors_empty: Vec<Vector<Mod7>> = vec![];
    let quotient_basis_empty_sub = compute_quotient_basis(&subspace_vectors_empty, &space_vectors);
    assert!(
      quotient_basis_empty_sub.is_empty(),
      "Quotient basis for zero-dimensional vectors (empty subspace) should be empty"
    );
  }

  #[test]
  fn test_quotient_all_zero_vectors_of_some_dimension() {
    // V = span{[0,0], [0,0]}, U = span{[0,0]}
    // V/U should be empty
    let u1_zero_vec = Vector::from(vec![Mod7::new(0), Mod7::new(0)]);
    let v1_zero_vec = Vector::from(vec![Mod7::new(0), Mod7::new(0)]);
    let v2_zero_vec = Vector::from(vec![Mod7::new(0), Mod7::new(0)]);

    let subspace_vectors = vec![u1_zero_vec.clone()];
    let space_vectors = vec![v1_zero_vec.clone(), v2_zero_vec.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(quotient_basis.is_empty(), "Quotient basis for all zero vectors should be empty");
  }
}
