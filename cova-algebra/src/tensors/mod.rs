//! Tensor utilities and linear-algebra helpers built on top of the [`nalgebra`] crate.
//!
//! This sub-module re-exports the whole of `nalgebra` so that downstream crates can simply do
//!
//! ```rust
//! use cova_algebra::tensors::*;
//! ```
//!
//! and immediately gain access to `nalgebra`'s `Matrix`, `Vector` and fixed-size aliases without
//! having to depend on the library directly.  On top of that it provides a handful of
//! **generic, field-agnostic** helpers that work for *any* type implementing
//! [`crate::rings::Field`]:
//!
//! * `compute_quotient_basis` – extracts a basis of the quotient space `V / U`.
//! * `image` / `kernel` – basis of a matrix' image / kernel computed via a reduced row–echelon
//!   transform written purely in terms of field operations.
//! * `MatrixBuilder` – a tiny builder-pattern wrapper that makes it ergonomic to write the small
//!   block matrices that show up in tests and in the Sheaf implementation.
//! * An implementation of the [`crate::category::Category`] trait for the dynamic column vector
//!   type `DVector<F>` so that *vectors are objects and matrices are morphisms*.
//!
//! None of these helpers require the scalar type `F` to implement `ClosedAdd`, `ClosedMul`, … and
//! therefore remain usable for finite-field types such as the `Mod7` fixture used throughout the
//! test-suite.

pub use nalgebra::*;
use num_traits::{One, Zero};

/// Computes the **Reduced Row-Echelon Form** (RREF) of the given dynamic matrix using a
/// plain-Rust Gauss–Jordan elimination that only requires the scalar type `F` to implement the
/// [`crate::rings::Field`] trait.
///
/// The routine returns a tuple `(rref, pivots)` where:
/// * `rref`   – a fresh `DMatrix<F>` in reduced row-echelon form.
/// * `pivots` – indices of the pivot columns (one index per pivot row) of the *original* matrix.
///
/// The algorithm works for *any* field (finite or infinite) because it never relies on floating-
/// point specific operations such as ε-pivoting.  It runs in-place on a clone of the input so the
/// original matrix is left untouched.
pub fn rref_with_pivots<F>(m: &DMatrix<F>) -> (DMatrix<F>, Vec<usize>)
where F: crate::rings::Field + Copy + Zero + PartialEq {
  let mut mat = m.clone();
  let nrows = mat.nrows();
  let ncols = mat.ncols();

  let mut pivot_cols = Vec::new();
  let mut current_row = 0usize;

  for col in 0..ncols {
    // Search a pivot row.
    if let Some(pivot_row) = (current_row..nrows).find(|&r| !mat[(r, col)].is_approx_zero()) {
      // Move it to the current row.
      if pivot_row != current_row {
        mat.swap_rows(pivot_row, current_row);
      }

      // Normalize the pivot row so that the pivot element is 1.
      let pivot_val = mat[(current_row, col)];
      let inv = pivot_val.multiplicative_inverse();
      for c in col..ncols {
        mat[(current_row, c)] *= inv;
      }

      // Eliminate this column from the other rows.
      for r in 0..nrows {
        if r == current_row {
          continue;
        }
        let factor = mat[(r, col)];
        if factor.is_zero() {
          continue;
        }
        for c in col..ncols {
          mat[(r, c)] = mat[(r, c)] - factor * mat[(current_row, c)];
        }
      }

      pivot_cols.push(col);
      current_row += 1;
      if current_row == nrows {
        break;
      }
    }
  }

  (mat, pivot_cols)
}

/// Returns a basis of the quotient space **V / U**.
///
/// * `subspace_vectors` – a slice of column vectors forming a basis of the sub-space **U**.
/// * `space_vectors`    – a slice of column vectors spanning the ambient space **V** (the caller
///   must ensure `U ⊆ V`).
///
/// The function concatenates the vectors `U ∪ V`, computes the RREF, and picks the pivot columns
/// that originate from the `space_vectors` slice – i.e. those columns that are linearly
/// independent *modulo* **U**.  The returned `Vec<DVector<F>>` therefore forms a basis of the
/// quotient space.
///
/// All vectors must share the same dimension; this is asserted in debug builds.
///
/// The routine is completely generic over the field `F`.
pub fn compute_quotient_basis<F: crate::rings::Field + Copy>(
  subspace_vectors: &[DVector<F>],
  space_vectors: &[DVector<F>],
) -> Vec<DVector<F>> {
  if space_vectors.is_empty() {
    return Vec::new();
  }

  // Determine the common dimension of all vectors from the first space vector.
  // All vectors (subspace and space) must have this same dimension.
  let expected_num_rows = space_vectors[0].len();

  // Verify all subspace vectors match this dimension.
  for (idx, vec) in subspace_vectors.iter().enumerate() {
    assert!(
      (vec.len() == expected_num_rows),
      "Subspace vector at index {} has dimension {} but expected {}",
      idx,
      vec.len(),
      expected_num_rows
    );
  }
  // Verify all other space vectors match this dimension.
  for (idx, vec) in space_vectors.iter().skip(1).enumerate() {
    assert!(
      (vec.len() == expected_num_rows),
      "Space vector at index {} (after first) has dimension {} but expected {}",
      idx + 1, // adjust index because of skip(1)
      vec.len(),
      expected_num_rows
    );
  }

  // Create matrix from columns of subspace + space.
  let mut all_columns = Vec::new();
  all_columns.extend_from_slice(subspace_vectors);
  all_columns.extend_from_slice(space_vectors);

  let matrix = DMatrix::<F>::from_columns(&all_columns);
  let (_, pivots) = rref_with_pivots(&matrix);
  let pivot_cols_set: std::collections::HashSet<usize> = pivots.into_iter().collect();

  let mut quotient_basis: Vec<DVector<F>> = Vec::new();
  let num_subspace_cols = subspace_vectors.len();

  for (i, original_space_vector) in space_vectors.iter().enumerate() {
    let augmented_matrix_col_idx = num_subspace_cols + i;
    if pivot_cols_set.contains(&augmented_matrix_col_idx) {
      quotient_basis.push(original_space_vector.clone());
    }
  }

  quotient_basis
}

/// Computes a **basis of the column space** (image) of `matrix`.
///
/// The implementation is field-agnostic: it calls [`rref_with_pivots`] and returns the original
/// columns that correspond to the pivot indices.
///
/// The resulting vectors are returned *in the order in which the pivots appear* in the original
/// matrix, which makes the function useful for deterministic test fixtures.
pub fn image<F>(matrix: &DMatrix<F>) -> Vec<DVector<F>>
where F: crate::rings::Field + Copy + Zero + PartialEq {
  let (_, pivot_cols) = rref_with_pivots(matrix);

  pivot_cols.into_iter().map(|col_idx| matrix.column(col_idx).into_owned()).collect()
}

/// Computes a **basis of the kernel** (null-space) of `matrix`.
///
/// A standard Gauss–Jordan solve is performed on an identity-augmented matrix to express the free
/// variables in terms of the pivots; one basis vector per free column is produced.  The method is
/// generic over any field `F` and does *not* require floating-point capabilities.
pub fn kernel<F>(matrix: &DMatrix<F>) -> Vec<DVector<F>>
where F: crate::rings::Field + ApproxZero + PartialEq {
  let ncols = matrix.ncols();
  if ncols == 0 {
    return Vec::new();
  }

  let nrows = matrix.nrows();
  if nrows == 0 {
    // Entire space is the kernel.
    return (0..ncols)
      .map(|i| {
        let mut comps = vec![F::zero(); ncols];
        comps[i] = F::one();
        DVector::from_row_slice(&comps)
      })
      .collect();
  }

  let (rref, pivot_cols) = rref_with_pivots(matrix);

  let mut is_pivot = vec![false; ncols];
  for &c in &pivot_cols {
    if c < ncols {
      is_pivot[c] = true;
    }
  }

  let free_cols: Vec<usize> = (0..ncols).filter(|&j| !is_pivot[j]).collect();

  let mut basis = Vec::new();
  for &free in &free_cols {
    let mut comps = vec![F::zero(); ncols];
    comps[free] = F::one();

    for (row_idx, &pivot_col) in pivot_cols.iter().enumerate() {
      let coeff = rref[(row_idx, free)];
      if !coeff.is_approx_zero() {
        comps[pivot_col] = -coeff;
      }
    }

    basis.push(DVector::from_row_slice(&comps));
  }

  basis
}

/// A **very** small builder-pattern utility aimed at test code and at places where manually
/// writing `DMatrix` literals would be unbearably verbose.
///
/// Example – create a 2 × 3 matrix by rows:
/// ```rust
/// # use cova_algebra::tensors::MatrixBuilder;
/// let m = MatrixBuilder::new().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).build();
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 3);
/// ```
///
/// The builder can also operate in *column* mode by calling `column` first.  Mixing `row` and
/// `column` calls will panic at runtime because the resulting layout would be ambiguous.
pub struct MatrixBuilder<F: crate::rings::Field> {
  mode: BuilderMode,
  data: Vec<Vec<F>>, // each inner Vec is a row or column depending on `mode`.
}

/// Internal state machine used by `MatrixBuilder` to ensure we do not mix row- and column-wise
/// initialisation.
enum BuilderMode {
  None,
  Row,
  Column,
}

impl<F: crate::rings::Field> MatrixBuilder<F> {
  /// Creates a fresh, empty `MatrixBuilder` in the *neutral* state – no rows/columns have been
  /// supplied yet so the builder does not know whether you intend to initialise by rows or by
  /// columns.
  ///
  /// The first call to either [`MatrixBuilder::row`] or [`MatrixBuilder::column`] fixes the mode;
  /// subsequent calls must use the same method or the builder will panic to avoid producing an
  /// ill-defined matrix.
  ///
  /// # Example
  /// ```rust
  /// # use cova_algebra::{tensors::MatrixBuilder, prelude::*};
  /// let m = MatrixBuilder::new().column([1.0, 0.0]).column([0.0, 1.0]).build();
  /// assert_eq!(m.nrows(), 2);
  /// assert_eq!(m.ncols(), 2);
  /// ```
  #[must_use]
  pub const fn new() -> Self { Self { mode: BuilderMode::None, data: Vec::new() } }

  /// Pushes a **column** into the matrix being built.
  ///
  /// All columns *must* have the same length – this is checked at `build` time.
  pub fn column<I>(mut self, values: I) -> Self
  where I: IntoIterator<Item = F> {
    let vals: Vec<F> = values.into_iter().collect();
    if matches!(self.mode, BuilderMode::Row) {
      panic!("MatrixBuilder: cannot mix row and column calls");
    }
    self.mode = BuilderMode::Column;
    self.data.push(vals);
    self
  }

  /// Pushes a **row** into the matrix being built.
  ///
  /// All rows *must* have the same length – this is checked at `build` time.
  pub fn row<I>(mut self, values: I) -> Self
  where I: IntoIterator<Item = F> {
    let vals: Vec<F> = values.into_iter().collect();
    if matches!(self.mode, BuilderMode::Column) {
      panic!("MatrixBuilder: cannot mix row and column calls");
    }
    self.mode = BuilderMode::Row;
    self.data.push(vals);
    self
  }

  /// Finalises the builder and produces an owned `DMatrix<F>`.
  ///
  /// If no `row`/`column` was ever supplied the resulting matrix is empty (0 × 0).
  pub fn build(self) -> DMatrix<F> {
    match self.mode {
      BuilderMode::None => DMatrix::zeros(0, 0),
      BuilderMode::Column => {
        // All columns must have same length.
        let ncols = self.data.len();
        if ncols == 0 {
          return DMatrix::zeros(0, 0);
        }
        let nrows = self.data[0].len();
        assert!(self.data.iter().all(|c| c.len() == nrows), "All columns must have same length");
        let vectors: Vec<DVector<F>> =
          self.data.into_iter().map(|v| DVector::from_row_slice(&v)).collect();
        DMatrix::from_columns(&vectors)
      },
      BuilderMode::Row => {
        let nrows = self.data.len();
        if nrows == 0 {
          return DMatrix::zeros(0, 0);
        }
        let ncols = self.data[0].len();
        assert!(self.data.iter().all(|r| r.len() == ncols), "All rows must have same length");
        let vectors: Vec<RowDVector<F>> =
          self.data.into_iter().map(|v| RowDVector::from_row_slice(&v)).collect();
        DMatrix::from_rows(&vectors)
      },
    }
  }
}

impl<F: crate::rings::Field> Default for MatrixBuilder<F> {
  fn default() -> Self { Self::new() }
}

use crate::{arithmetic::ApproxZero, category::Category};

impl<F> Category for DVector<F>
where F: crate::rings::Field + Copy + Zero + One
{
  type Morphism = DMatrix<F>;

  // Compose two linear maps represented by matrices via a simple, generic
  // O(n^3) multiplication that only relies on the `Field` operations instead
  // of nalgebra's `ClosedMul`/`ClosedAdd` traits (which our custom finite
  // fields don't implement).
  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism {
    assert_eq!(f.ncols(), g.nrows(), "Incompatible dimensions for composition");
    let m = f.nrows();
    let n = g.ncols();
    let k = f.ncols();
    let mut result = DMatrix::<F>::zeros(m, n);
    for i in 0..m {
      for j in 0..n {
        let mut acc = F::zero();
        for p in 0..k {
          acc += f[(i, p)] * g[(p, j)];
        }
        result[(i, j)] = acc;
      }
    }
    result
  }

  fn identity(a: Self) -> Self::Morphism {
    let n = a.len();
    let mut id = DMatrix::<F>::zeros(n, n);
    for i in 0..n {
      id[(i, i)] = F::one();
    }
    id
  }

  fn apply(f: Self::Morphism, x: Self) -> Self {
    assert_eq!(f.ncols(), x.len(), "Matrix–vector dimension mismatch");
    let m = f.nrows();
    let n = f.ncols();
    let mut out = Self::zeros(m);
    for i in 0..m {
      let mut acc = F::zero();
      for j in 0..n {
        acc += f[(i, j)] * x[j];
      }
      out[i] = acc;
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::{compute_quotient_basis, image, kernel};
  use crate::{
    fixtures::Mod7,
    tensors::{DMatrix, DVector},
  };

  #[test]
  fn test_quotient_simple_span() {
    // V = span{[1,0,0], [0,1,0]}, U = span{[1,0,0]}
    // V/U should be span{[0,1,0]}
    let u1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0), Mod7::new(0)]);
    let v_in_u = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0), Mod7::new(0)]);
    let v_new = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1), Mod7::new(0)]);

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
    let u1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]);
    let u2 = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1)]);
    // space_vectors are the same as subspace_vectors
    let space_vectors = vec![u1.clone(), u2.clone()];
    let subspace_vectors = vec![u1, u2];

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
    let v1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]);
    let v2 = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1)]);

    let subspace_vectors: Vec<DVector<Mod7>> = vec![];
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
    let u1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]);
    let v_in_u = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]); // Effectively in U
    let v_dependent_on_u = DVector::<Mod7>::from_row_slice(&[Mod7::new(2), Mod7::new(0)]); // 2*u1
    let v_new_independent = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1)]);

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
    let u1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0), Mod7::new(0)]);

    let v1_new = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1), Mod7::new(0)]);
    let v2_dependent_on_v1 =
      DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(2), Mod7::new(0)]);
    let v3_new = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(0), Mod7::new(1)]);

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
    let u1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]);
    let subspace_vectors = vec![u1];
    let space_vectors: Vec<DVector<Mod7>> = vec![];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(quotient_basis.is_empty(), "Quotient basis should be empty if space_vectors is empty");
  }

  #[test]
  fn test_quotient_zero_dimensional_vectors() {
    let u1_zero_dim = DVector::<Mod7>::zeros(0);
    let v1_zero_dim = DVector::<Mod7>::zeros(0);
    let v2_zero_dim = DVector::<Mod7>::zeros(0);

    let subspace_vectors = vec![u1_zero_dim];
    let space_vectors = vec![v1_zero_dim, v2_zero_dim];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(
      quotient_basis.is_empty(),
      "Quotient basis for zero-dimensional vectors should be empty"
    );

    // Case: subspace empty, space has 0-dim vectors
    let subspace_vectors_empty: Vec<DVector<Mod7>> = vec![];
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
    let u1_zero_vec = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(0)]);
    let v1_zero_vec = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(0)]);
    let v2_zero_vec = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(0)]);

    let subspace_vectors = vec![u1_zero_vec.clone()];
    let space_vectors = vec![v1_zero_vec.clone(), v2_zero_vec.clone()];

    let quotient_basis = compute_quotient_basis(&subspace_vectors, &space_vectors);
    assert!(quotient_basis.is_empty(), "Quotient basis for all zero vectors should be empty");
  }

  #[test]
  fn test_image_simple() {
    // Matrix with dependent first two columns.
    let col1 = DVector::<Mod7>::from_row_slice(&[Mod7::new(1), Mod7::new(0)]);
    let col2 = DVector::<Mod7>::from_row_slice(&[Mod7::new(2), Mod7::new(0)]); // 2 * col1
    let col3 = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1)]);

    let m = DMatrix::<Mod7>::from_columns(&[col1.clone(), col2.clone(), col3.clone()]);
    let img_basis = image(&m);

    assert_eq!(img_basis.len(), 2);
    assert!(img_basis.contains(&col1) && img_basis.contains(&col3));
  }

  #[test]
  fn test_kernel_simple() {
    // 1 x 2 matrix [1 0]; kernel should be all vectors of the form (0, t).
    let m = DMatrix::<Mod7>::from_row_slice(1, 2, &[Mod7::new(1), Mod7::new(0)]);
    let ker_basis = kernel(&m);

    assert_eq!(ker_basis.len(), 1);
    let expected = DVector::<Mod7>::from_row_slice(&[Mod7::new(0), Mod7::new(1)]);
    assert_eq!(ker_basis[0], expected);
  }
}
