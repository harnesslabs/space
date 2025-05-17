//! Linear algebra functions for computations over fields.
//!
//! This module provides implementations of common linear algebra algorithms,
//! particularly focusing on Gaussian elimination. These functions are generic
//! and can operate on matrices whose elements are from any type that implements
//! the [`Field`] trait and is [`Copy`].
//!
//! Common field types used with this module include:
//! - Standard floating-point numbers: `f64`, `f32`.
//! - Boolean values: [`Boolean`](crate::arithmetic::Boolean), representing the field
//!   $\\mathbb{Z}/2\\mathbb{Z}$.
//! - Modular arithmetic types: For example, types representing $\\mathbb{Z}/p\\mathbb{Z}$ for a
//!   prime $p$, such as the `Mod7` type used in tests (generated via macros like `modular!` and
//!   `prime_field!`).
//!
//! The primary functions provided are:
//! - [`row_gaussian_elimination`]: Transforms a matrix into its row echelon form (specifically,
//!   reduced row echelon form).
//! - [`column_gaussian_elimination`]: Transforms a matrix into a column echelon form using row
//!   operations.
use crate::ring::Field;

/// Performs Gaussian elimination on a matrix to bring it to column echelon form
/// using row operations. The matrix is modified in place.
///
/// The "column echelon form" achieved here means that:
/// 1. Each pivot element is 1.
/// 2. All other entries in a pivot column (both above and below the pivot) are 0.
/// 3. Pivot columns are to the left of non-pivot columns within the scope of rows processed so far.
///
/// This function iterates through columns, identifies a pivot element in the current column (at or
/// below the current pivot row), swaps rows if necessary to bring the pivot to the `pivot_row`,
/// normalizes the `pivot_row` so the pivot element becomes 1, and then uses row operations to
/// eliminate all other non-zero elements in the current pivot column.
///
/// # Arguments
///
/// * `matrix`: A mutable slice of vectors, representing the matrix. Each [`Vec<F>`] is a row. The
///   matrix elements must be of a type `F` that implements [`Field`] and [`Copy`].
///
/// # Returns
///
/// The rank of the matrix, which is the number of pivot columns found.
///
/// # Examples
///
/// ```
/// use harness_algebra::ring::Field; // For trait bounds
/// use harness_algebra::{
///   arithmetic::{Boolean, One, Zero},
///   linear::column_gaussian_elimination,
/// }; // For Boolean example
///
/// // Example with Boolean (representing Z/2Z)
/// let mut matrix_z2 = vec![vec![Boolean(true), Boolean(true), Boolean(false)], vec![
///   Boolean(true),
///   Boolean(false),
///   Boolean(true),
/// ]];
/// let rank_z2 = column_gaussian_elimination(&mut matrix_z2);
/// assert_eq!(rank_z2, 2);
/// // Expected column echelon form (one possibility):
/// // [[1, 0, 1],
/// //  [0, 1, 1]]
/// assert_eq!(matrix_z2, vec![vec![Boolean(true), Boolean(false), Boolean(true)], vec![
///   Boolean(false),
///   Boolean(true),
///   Boolean(true)
/// ],]);
///
/// // Example with f64
/// let mut matrix_f64 = vec![vec![1.0_f64, 2.0, 1.0], vec![2.0, 4.0, 0.0], vec![3.0, 6.0, 0.0]];
/// let rank_f64 = column_gaussian_elimination(&mut matrix_f64);
/// assert_eq!(rank_f64, 2);
/// // Expected column echelon form (one possibility for f64):
/// // [[1.0, 2.0, 0.0],
/// //  [0.0, 0.0, 1.0],
/// //  [0.0, 0.0, 0.0]]
/// // Note: The exact form can depend on pivot choices if multiple exist.
/// // For the given example and implementation:
/// // Pivot (0,0) -> 1.0. Row 0: [1,2,1].
/// //   R1 -= 2*R0 => [0,0,-2]
/// //   R2 -= 3*R0 => [0,0,-3]
/// // Matrix: [[1,2,1],[0,0,-2],[0,0,-3]]
/// // Pivot (0,0) is 1. Clear column 0 for other rows (already done). pivot_row=1, rank=1.
/// // Next column (j=1). Pivot search from row 1. matrix[1][1]=0, matrix[2][1]=0. No pivot.
/// // Next column (j=2). Pivot search from row 1. matrix[1][2]=-2. Pivot is -2 at (1,2).
/// //   Swap row 1 with pivot_row (1) (no change).
/// //   Normalize row 1: R1 /= -2 => [0,0,1]
/// //   Matrix: [[1,2,1],[0,0,1],[0,0,-3]]
/// //   Pivot (1,2) is 1. Clear column 2 for other rows.
/// //     R0 -= 1*R1 => [1,2,0]
/// //     R2 -= -3*R1 => [0,0,0]
/// // Final Matrix: [[1,2,0],[0,0,1],[0,0,0]]
/// assert!((matrix_f64[0][0] - 1.0).abs() < 1e-9);
/// assert!((matrix_f64[0][1] - 2.0).abs() < 1e-9);
/// assert!(matrix_f64[0][2].abs() < 1e-9);
/// assert!(matrix_f64[1][0].abs() < 1e-9);
/// assert!(matrix_f64[1][1].abs() < 1e-9);
/// assert!((matrix_f64[1][2] - 1.0).abs() < 1e-9);
/// assert!(matrix_f64[2][0].abs() < 1e-9);
/// assert!(matrix_f64[2][1].abs() < 1e-9);
/// assert!(matrix_f64[2][2].abs() < 1e-9);
/// ```
pub fn column_gaussian_elimination<F: Field + Copy>(matrix: &mut [Vec<F>]) -> usize {
  if matrix.is_empty() || matrix[0].is_empty() {
    return 0;
  }
  let rows = matrix.len();
  let cols = matrix[0].len();
  let mut pivot_row = 0;
  let mut rank = 0;

  for j in 0..cols {
    // Iterate through columns (potential pivot columns)
    if pivot_row >= rows {
      break;
    }
    let mut i = pivot_row;
    while i < rows && matrix[i][j].is_zero() {
      i += 1;
    }

    if i < rows {
      // Found a pivot in this column at matrix[i][j]
      // Swap row i with pivot_row to bring pivot to matrix[pivot_row][j]
      if i != pivot_row {
        for col_idx_swap in j..cols {
          let temp = matrix[i][col_idx_swap];
          matrix[i][col_idx_swap] = matrix[pivot_row][col_idx_swap];
          matrix[pivot_row][col_idx_swap] = temp;
        }
      }

      // matrix[pivot_row][j] is now the pivot element.
      let pivot_val = matrix[pivot_row][j];

      // Normalize the pivot row so that matrix[pivot_row][j] becomes 1.
      // This is crucial for general fields.
      if !pivot_val.is_zero() {
        // Pivot val by definition here should not be zero
        let inv_pivot_val = pivot_val.multiplicative_inverse();
        for col_idx_norm in j..cols {
          matrix[pivot_row][col_idx_norm] *= inv_pivot_val;
        }
      }
      // Now matrix[pivot_row][j] is 1 (if pivot_val was not zero).

      // Eliminate other non-zero entries in the current column j (i.e., make matrix[k][j] = 0 for k
      // != pivot_row) by row operations: R_k = R_k - factor * R_pivot_row.
      for k in 0..rows {
        if k != pivot_row {
          let factor = matrix[k][j]; // This is the value to eliminate in column j, row k.
                                     // Since matrix[pivot_row][j] is now 1, this factor is correct.
          if !factor.is_zero() {
            for col_idx_elim in j..cols {
              // Iterate from the pivot column to the right
              let term_to_subtract = factor * matrix[pivot_row][col_idx_elim];
              matrix[k][col_idx_elim] -= term_to_subtract;
            }
          }
        }
      }
      pivot_row += 1;
      rank += 1;
    }
  }
  rank
}

/// Performs Gaussian elimination on a matrix over Z2 to bring it to row echelon form.
/// The matrix is modified in place.
///
/// The reduced row echelon form satisfies the following properties:
/// 1. If a row has non-zero entries, its first non-zero entry (pivot) is 1.
/// 2. All other entries in a pivot column are 0.
/// 3. Rows consisting entirely of zeros are at the bottom of the matrix.
/// 4. For any two pivot rows, the pivot in the upper row is to the left of the pivot in the lower
///    row.
///
/// # Arguments
///
/// * `matrix`: A mutable slice of vectors, representing the matrix. Each [`Vec<F>`] is a row. The
///   matrix elements must be of a type `F` that implements [`Field`] and [`Copy`].
///
/// # Returns
///
/// A tuple `(rank, pivot_columns_indices)`:
/// * `rank`: The rank of the matrix (number of non-zero rows in RREF, or number of pivots).
/// * `pivot_columns_indices`: A vector containing the column indices of the pivot elements, ordered
///   by the row in which they appear.
///
/// # Examples
///
/// ```
/// use harness_algebra::ring::Field; // For trait bounds
/// use harness_algebra::{
///   arithmetic::{Boolean, One, Zero},
///   linear::row_gaussian_elimination,
/// }; // For Boolean example
///
/// // Example with Boolean (representing Z/2Z)
/// let mut matrix_z2 = vec![
///   vec![Boolean(true), Boolean(true), Boolean(false)], // [1, 1, 0]
///   vec![Boolean(true), Boolean(false), Boolean(true)], // [1, 0, 1]
/// ];
/// let (rank_z2, pivots_z2) = row_gaussian_elimination(&mut matrix_z2);
/// assert_eq!(rank_z2, 2);
/// assert_eq!(pivots_z2, vec![0, 1]); // Pivots at (0,0) and (1,1) after transformation
///                                    // R1 = R1 (no change initially as matrix[0][0] is pivot)
///                                    // R2 = R2 + R1 = [1,0,1] + [1,1,0] = [0,1,1]
///                                    // Matrix: [[1,1,0], [0,1,1]]
///                                    // Pivot for R2 is matrix[1][1]=1.
///                                    // R1 = R1 + R2 = [1,1,0] + [0,1,1] = [1,0,1]
///                                    // Final RREF:
///                                    // [[1, 0, 1],
///                                    //  [0, 1, 1]]
/// assert_eq!(matrix_z2, vec![vec![Boolean(true), Boolean(false), Boolean(true)], vec![
///   Boolean(false),
///   Boolean(true),
///   Boolean(true)
/// ],]);
///
/// // Example with f64
/// let mut matrix_f64 =
///   vec![vec![1.0_f64, 2.0, -1.0, 3.0], vec![2.0, 4.0, -2.0, 7.0], vec![0.0, 1.0, 2.0, 1.0]];
/// let (rank_f64, pivots_f64) = row_gaussian_elimination(&mut matrix_f64);
/// assert_eq!(rank_f64, 3);
/// // Expected pivot columns might be [0, 1, 3] or similar depending on exact RREF steps.
/// // Let's check the RREF form.
/// // Expected RREF (one possibility):
/// // [[1.0, 0.0, -5.0, 0.0],
/// //  [0.0, 1.0,  2.0, 0.0],
/// //  [0.0, 0.0,  0.0, 1.0]]
/// // This would imply pivots_f64 = vec![0,1,3]
/// assert_eq!(pivots_f64, vec![0, 1, 3]);
///
/// let expected_rref_f64 =
///   vec![vec![1.0, 0.0, -5.0, 0.0], vec![0.0, 1.0, 2.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];
/// for i in 0..matrix_f64.len() {
///   for j in 0..matrix_f64[i].len() {
///     assert!((matrix_f64[i][j] - expected_rref_f64[i][j]).abs() < 1e-9);
///   }
/// }
/// ```
pub fn row_gaussian_elimination<F: Field + Copy>(matrix: &mut [Vec<F>]) -> (usize, Vec<usize>) {
  if matrix.is_empty() || matrix[0].is_empty() {
    return (0, Vec::new());
  }
  let rows = matrix.len();
  let cols = matrix[0].len();
  let mut lead = 0; // current pivot column
  let mut rank = 0;
  let mut pivot_cols = Vec::new();

  for r in 0..rows {
    if lead >= cols {
      break;
    }
    let mut i = r;
    while matrix[i][lead].is_zero() {
      i += 1;
      if i == rows {
        i = r;
        lead += 1;
        if lead == cols {
          return (rank, pivot_cols);
        }
      }
    }
    matrix.swap(i, r);
    pivot_cols.push(lead);

    let pivot_val = matrix[r][lead];
    // For a field, a non-zero pivot_val is expected here.
    // The Field trait should provide inverse. Panicking if not found for a non-zero element is
    // acceptable.
    let inv_pivot = pivot_val.multiplicative_inverse();

    // Normalize pivot row: matrix[r][j] = matrix[r][j] * inv_pivot
    for j in lead..cols {
      matrix[r][j] *= inv_pivot;
    }

    // Eliminate other rows: matrix[i] = matrix[i] - factor * matrix[r]
    for i_row in 0..rows {
      if i_row != r {
        let factor = matrix[i_row][lead]; // factor is F, which is Copy
        if !factor.is_zero() {
          for j_col in lead..cols {
            let term = factor * matrix[r][j_col];
            matrix[i_row][j_col] -= term;
          }
        }
      }
    }
    lead += 1;
    rank += 1;
  }
  (rank, pivot_cols)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{arithmetic::Boolean, modular, prime_field};

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);

  // Helper function to create a matrix of Booleans
  fn bool_matrix(data: Vec<Vec<bool>>) -> Vec<Vec<Boolean>> {
    data.into_iter().map(|row| row.into_iter().map(Boolean).collect()).collect()
  }

  // Helper function to create a matrix of ModN<7>
  fn mod7_matrix(data: Vec<Vec<u32>>) -> Vec<Vec<Mod7>> {
    data.into_iter().map(|row| row.into_iter().map(Mod7::new).collect()).collect()
  }

  // Tests for row_gaussian_elimination with f64
  #[test]
  fn test_row_gaussian_elimination_f64_empty() {
    let mut matrix: Vec<Vec<f64>> = vec![];
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);
    assert!(pivot_cols.is_empty());

    let mut matrix_empty_rows: Vec<Vec<f64>> = vec![vec![]];
    let (rank_empty_rows, pivot_cols_empty_rows) = row_gaussian_elimination(&mut matrix_empty_rows);
    assert_eq!(rank_empty_rows, 0);
    assert!(pivot_cols_empty_rows.is_empty());
  }

  #[test]
  fn test_row_gaussian_elimination_f64_identity() {
    let mut matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    let (rank, _pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    // Matrix should be unchanged as it's already in RRE form
    assert_eq!(matrix, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0],]);
  }

  #[test]
  fn test_row_gaussian_elimination_f64_simple() {
    let mut matrix = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 1);
    assert_eq!(pivot_cols, vec![0]);
    // Expected RRE form (or one valid RRE form)
    // [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
    for row in &matrix {
      row.iter().for_each(|x: &f64| assert!((x.abs() < 1e-9) || (x - x.round()).abs() < 1e-9));
    } // Check for near zero or integer for simplicity
    assert!((matrix[0][0] - 1.0f64).abs() < 1e-9);
    assert!((matrix[0][1] - 2.0f64).abs() < 1e-9);
    assert!((matrix[0][2] - 3.0f64).abs() < 1e-9);
    assert!(matrix[1][0].abs() < 1e-9);
    assert!(matrix[1][1].abs() < 1e-9);
    assert!(matrix[1][2].abs() < 1e-9);
  }

  #[test]
  fn test_row_gaussian_elimination_f64_swap_and_reduce() {
    let mut matrix = vec![vec![0.0, 1.0, 2.0], vec![1.0, -1.0, 1.0], vec![2.0, -2.0, 3.0]];
    let (rank, _pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    // Expected RRE form:
    // [[1.0, 0.0, 0.0],
    //  [0.0, 1.0, 0.0],
    //  [0.0, 0.0, 1.0]]
    assert!((matrix[0][0] - 1.0f64).abs() < 1e-9);
    assert!(matrix[0][1].abs() < 1e-9);
    assert!(matrix[0][2].abs() < 1e-9);
    assert!(matrix[1][0].abs() < 1e-9);
    assert!((matrix[1][1] - 1.0f64).abs() < 1e-9);
    assert!(matrix[1][2].abs() < 1e-9);
    assert!(matrix[2][0].abs() < 1e-9);
    assert!(matrix[2][1].abs() < 1e-9);
    assert!((matrix[2][2] - 1.0f64).abs() < 1e-9);
  }

  // Tests for row_gaussian_elimination with Boolean
  #[test]
  fn test_row_gaussian_elimination_boolean_empty() {
    let mut matrix: Vec<Vec<Boolean>> = vec![];
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);
    assert!(pivot_cols.is_empty());

    let mut matrix_empty_rows = bool_matrix(vec![vec![]]);
    let (rank_empty_rows, pivot_cols_empty_rows) = row_gaussian_elimination(&mut matrix_empty_rows);
    assert_eq!(rank_empty_rows, 0);
    assert!(pivot_cols_empty_rows.is_empty());
  }

  #[test]
  fn test_row_gaussian_elimination_boolean_identity() {
    let mut matrix = bool_matrix(vec![vec![true, false, false], vec![false, true, false], vec![
      false, false, true,
    ]]);
    let (rank, _pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    assert_eq!(
      matrix,
      bool_matrix(vec![vec![true, false, false], vec![false, true, false], vec![
        false, false, true
      ],])
    );
  }

  #[test]
  fn test_row_gaussian_elimination_boolean_simple() {
    let mut matrix = bool_matrix(vec![
      vec![true, true, false], // R1
      vec![true, false, true], // R2
    ]);
    // R2 = R2 + R1 = [0, 1, 1]
    // Matrix becomes:
    // [1, 1, 0]
    // [0, 1, 1]
    // R1 = R1 + R2 = [1, 0, 1]
    // Matrix becomes:
    // [1, 0, 1]
    // [0, 1, 1]
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 2);
    assert_eq!(pivot_cols, vec![0, 1]);
    assert_eq!(matrix, bool_matrix(vec![vec![true, false, true], vec![false, true, true],]));
  }

  #[test]
  fn test_row_gaussian_elimination_boolean_dependent_rows() {
    let mut matrix = bool_matrix(vec![
      vec![true, true, false],
      vec![false, true, true],
      vec![true, false, true], // R1 + R2
    ]);
    // R3 = R3 + R1 = [0, 1, 1]
    // R3 = R3 + R2 = [0, 0, 0]
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 2);
    assert_eq!(pivot_cols, vec![0, 1]); // or [0,2] depending on choices, but with current code it's [0,1]
    assert_eq!(
      matrix,
      bool_matrix(vec![
        vec![true, false, true],   // R1' = R1 + R2
        vec![false, true, true],   // R2
        vec![false, false, false], // R3'
      ])
    );
  }

  // Tests for column_gaussian_elimination with f64
  #[test]
  fn test_column_gaussian_elimination_f64_empty() {
    let mut matrix: Vec<Vec<f64>> = vec![];
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);

    let mut matrix_empty_cols: Vec<Vec<f64>> = vec![vec![], vec![]];
    let rank_empty_cols = column_gaussian_elimination(&mut matrix_empty_cols);
    assert_eq!(rank_empty_cols, 0);
  }

  #[test]
  fn test_column_gaussian_elimination_f64_identity() {
    let mut matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    // Column Echelon form might vary slightly, this is one.
    // The function ensures pivots are 1 and entries below are 0.
    // It also clears above the pivot in the current column.
    // So identity should remain identity.
    assert_eq!(matrix, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0],]);
  }

  #[test]
  fn test_column_gaussian_elimination_f64_simple() {
    let mut matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
    // Expected column echelon form (one possibility):
    // [[1.0, 2.0],
    //  [0.0, 0.0],
    //  [0.0, 0.0]]
    // Rank is 1
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 1);
    // Check that the first column is a pivot column and the second is effectively zeroed
    // relative to the pivot structure.
    // After correct GE, matrix should be something like:
    // [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]
    assert!((matrix[0][0] - 1.0f64).abs() < 1e-9);
    assert!((matrix[0][1] - 2.0f64).abs() < 1e-9); // R0 can remain [1,2] if C2 depends on C1
    (1..matrix.len()).for_each(|r_idx| {
      assert!(matrix[r_idx][0].abs() < 1e-9);
      assert!(matrix[r_idx][1].abs() < 1e-9);
    });

    // The previous, more complex assertions for this test case regarding the matrix state
    // can be simplified if we just check the rank and general structure.
    // The primary check is that the rank is 1.
    // A more detailed check of the matrix state:
    let expected_matrix = [vec![1.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]];
    for r in 0..matrix.len() {
      for c in 0..matrix[0].len() {
        assert!(
          (matrix[r][c] - expected_matrix[r][c]).abs() < 1e-9,
          "Matrix entry ({r},{c}) mismatch",
        );
      }
    }
  }

  // Tests for column_gaussian_elimination with Boolean
  #[test]
  fn test_column_gaussian_elimination_boolean_empty() {
    let mut matrix: Vec<Vec<Boolean>> = vec![];
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);

    let mut matrix_empty_cols = bool_matrix(vec![vec![], vec![]]);
    let rank_empty_cols = column_gaussian_elimination(&mut matrix_empty_cols);
    assert_eq!(rank_empty_cols, 0);
  }

  #[test]
  fn test_column_gaussian_elimination_boolean_identity() {
    let mut matrix = bool_matrix(vec![vec![true, false, false], vec![false, true, false], vec![
      false, false, true,
    ]]);
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    assert_eq!(
      matrix,
      bool_matrix(vec![vec![true, false, false], vec![false, true, false], vec![
        false, false, true
      ],])
    );
  }

  #[test]
  fn test_column_gaussian_elimination_boolean_simple() {
    let mut matrix = bool_matrix(vec![
      vec![true, true],  // R1
      vec![true, false], // R2
    ]);
    // Col 0: Pivot at (0,0) is true.
    // Row 1 is not pivot row (k=1), matrix[1][0] is true.
    // matrix[1][0] = matrix[1][0] + matrix[0][0] = true + true = false
    // matrix[1][1] = matrix[1][1] + matrix[0][1] = false + true = true
    // Matrix becomes:
    // [true, true]
    // [false, true]
    // Pivot_row becomes 1, rank becomes 1.
    //
    // Col 1: Pivot search from row 1. matrix[1][1] is true. Pivot at (1,1).
    // Row 0 is not pivot row (k=0), matrix[0][1] is true.
    // matrix[0][1] = matrix[0][1] + matrix[1][1] = true + true = false
    // (matrix[0][0] is not affected as col_idx starts from j=1)
    // Correct loop for clearing: for col_idx in j..cols
    // So for Col 0 (j=0), pivot_row=0:
    //   k=1: matrix[1][0] != 0.
    //     matrix[1][0] = matrix[1][0] + matrix[0][0] = T + T = F
    //     matrix[1][1] = matrix[1][1] + matrix[0][1] = F + T = T
    //   Matrix: [[T,T], [F,T]]. pivot_row=1, rank=1.
    //
    // For Col 1 (j=1), pivot_row=1:
    //   Pivot search from row 1: matrix[1][1] is T. i=1.
    //   Swap not needed.
    //   k=0: matrix[0][1] != 0.
    //     matrix[0][1] = matrix[0][1] + matrix[1][1] = T + T = F
    //   Matrix: [[T,F], [F,T]]. pivot_row=2, rank=2.
    //
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 2);
    assert_eq!(matrix, bool_matrix(vec![vec![true, false], vec![false, true],]));
  }

  #[test]
  fn test_column_gaussian_elimination_boolean_dependent_cols() {
    let mut matrix = bool_matrix(vec![
      vec![true, true, false], // C1 C2 C3
      vec![true, false, true],
      vec![false, true, true],
    ]);
    // Expected: transform to [[T,F,F],[F,T,F],[F,F,T]] if full rank, or identify dependency.
    // Col 0 (j=0), pivot_row=0:
    //  Pivot matrix[0][0]=T.
    //  k=1 (row 1): matrix[1][0]=T.
    //   matrix[1][0] = m[1][0]+m[0][0] = T+T = F
    //   matrix[1][1] = m[1][1]+m[0][1] = F+T = T
    //   matrix[1][2] = m[1][2]+m[0][2] = T+F = T
    //  Matrix after processing row 1:
    //  [[T,T,F],
    //   [F,T,T],
    //   [F,T,T]]
    //  k=2 (row 2): matrix[2][0]=F. No change for row 2 based on matrix[2][0].
    //  pivot_row=1, rank=1.
    //
    // Col 1 (j=1), pivot_row=1:
    //  Pivot search from row 1: matrix[1][1]=T. i=1. No swap.
    //  k=0 (row 0): matrix[0][1]=T.
    //   matrix[0][1] = m[0][1]+m[1][1] = T+T = F
    //   matrix[0][2] = m[0][2]+m[1][2] = F+T = T
    //  Matrix after processing row 0:
    //  [[T,F,T],
    //   [F,T,T],
    //   [F,T,T]]
    //  k=2 (row 2): matrix[2][1]=T.
    //   matrix[2][1] = m[2][1]+m[1][1] = T+T = F
    //   matrix[2][2] = m[2][2]+m[1][2] = T+T = F
    //  Matrix after processing row 2:
    //  [[T,F,T],
    //   [F,T,T],
    //   [F,F,F]]
    //  pivot_row=2, rank=2.
    //
    // Resulting matrix:
    // [[T,F,T],
    //  [F,T,T],
    //  [F,F,F]]
    // Rank = 2.
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 2);
    assert_eq!(
      matrix,
      bool_matrix(vec![vec![true, false, true], vec![false, true, true], vec![
        false, false, false
      ],])
    );
  }

  // Tests for row_gaussian_elimination with ModN<7>
  #[test]
  fn test_row_gaussian_elimination_mod7_empty() {
    let mut matrix: Vec<Vec<Mod7>> = vec![];
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);
    assert!(pivot_cols.is_empty());

    let mut matrix_empty_rows = mod7_matrix(vec![vec![]]);
    let (rank_empty_rows, pivot_cols_empty_rows) = row_gaussian_elimination(&mut matrix_empty_rows);
    assert_eq!(rank_empty_rows, 0);
    assert!(pivot_cols_empty_rows.is_empty());
  }

  #[test]
  fn test_row_gaussian_elimination_mod7_identity() {
    let mut matrix = mod7_matrix(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    assert_eq!(pivot_cols, vec![0, 1, 2]);
    assert_eq!(matrix, mod7_matrix(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1],]));
  }

  #[test]
  fn test_row_gaussian_elimination_mod7_simple() {
    let mut matrix = mod7_matrix(vec![vec![1, 2, 3], vec![2, 4, 6]]);
    // R2 = R2 - 2*R1 = [2,4,6] - [2,4,6] = [0,0,0]
    // Expected:
    // [1, 2, 3]
    // [0, 0, 0]
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 1);
    assert_eq!(pivot_cols, vec![0]);
    assert_eq!(matrix, mod7_matrix(vec![vec![1, 2, 3], vec![0, 0, 0],]));
  }

  #[test]
  fn test_row_gaussian_elimination_mod7_swap_and_reduce() {
    let mut matrix = mod7_matrix(vec![
      vec![0, 1, 2], // R1
      vec![2, 1, 0], // R2
      vec![1, 3, 5], // R3
    ]);
    // Swap R1 and R2 (or R1 and R3). Let's say R1 <-> R2 (if pivot search picks R2[0]) or R1 <-> R3
    // If R1 <-> R3:
    // [1, 3, 5] (new R1)
    // [2, 1, 0] (R2)
    // [0, 1, 2] (new R3)
    // Normalize R1 (already is: 1*inv(1)=1)
    // R2 = R2 - 2*R1 = [2,1,0] - 2*[1,3,5] = [2,1,0] - [2,6,10] = [2,1,0] - [2,6,3] = [0, -5, -3] =
    // [0, 2, 4] Matrix:
    // [1, 3, 5]
    // [0, 2, 4]
    // [0, 1, 2]
    // Pivot for R2 is 2. Normalize R2: R2 = R2 * inv(2). inv(2) mod 7 is 4. (2*4=8=1 mod 7)
    // R2 = [0,2,4]*4 = [0,8,16] = [0,1,2]
    // Matrix:
    // [1, 3, 5]
    // [0, 1, 2]
    // [0, 1, 2]
    // Eliminate for R1: R1 = R1 - 3*R2 = [1,3,5] - 3*[0,1,2] = [1,3,5] - [0,3,6] = [1,0,-1] =
    // [1,0,6] Eliminate for R3: R3 = R3 - 1*R2 = [0,1,2] - [0,1,2] = [0,0,0]
    // Matrix:
    // [1, 0, 6]
    // [0, 1, 2]
    // [0, 0, 0]
    let (rank, pivot_cols) = row_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 2);
    assert_eq!(pivot_cols, vec![0, 1]); // Pivot columns are 0 and 1
    assert_eq!(matrix, mod7_matrix(vec![vec![1, 0, 6], vec![0, 1, 2], vec![0, 0, 0],]));
  }

  // Tests for column_gaussian_elimination with ModN<7>
  // As noted, column_gaussian_elimination's use of '+' for elimination is specific to char 2
  // fields. These tests will be very basic for ModN<7>.
  #[test]
  fn test_column_gaussian_elimination_mod7_identity() {
    let mut matrix = mod7_matrix(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 3);
    // For identity, the clearing logic using '+' does not change other rows/cols if they are 0.
    assert_eq!(matrix, mod7_matrix(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1],]));
  }

  #[test]
  fn test_column_gaussian_elimination_mod7_zero() {
    let mut matrix = mod7_matrix(vec![vec![0, 0], vec![0, 0]]);
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 0);
    assert_eq!(matrix, mod7_matrix(vec![vec![0, 0], vec![0, 0]]));
  }

  #[test]
  fn test_column_gaussian_elimination_mod7_single_element_col_pivot() {
    // This test highlights the issue with \'+\' for elimination in non-char 2 fields.
    let mut matrix = mod7_matrix(vec![vec![2], vec![3]]);
    // Expected after general field GE:
    // Pivot is 2 at (0,0). Normalize R0: R0 = R0 * inv(2) = R0 * 4 = [2*4] = [8] = [1]
    // Matrix: [[1],[3]]
    // Eliminate matrix[1][0]: factor = matrix[1][0] = 3.
    // R1 = R1 - 3*R0 = [3] - 3*[1] = [3] - [3] = [0]
    // Matrix: [[1],[0]]
    // Rank = 1
    let rank = column_gaussian_elimination(&mut matrix);
    assert_eq!(rank, 1);
    assert_eq!(matrix, mod7_matrix(vec![vec![Mod7::new(1).0], vec![Mod7::new(0).0]]));
  }
}
