//! # Dynamic Matrix Module
//!
//! This module provides a flexible implementation of matrices with dynamically determined
//! dimensions.
//!
//! ## Mathematical Background
//!
//! A matrix is a rectangular array of elements arranged in rows and columns. For a matrix $A$
//! with $m$ rows and $n$ columns, we write $A \in F^{m \times n}$ where $F$ is the field of
//! the matrix elements.
//!
//! ### Matrix Operations
//!
//! Matrices support various operations including:
//!
//! - **Transposition**: For a matrix $A$, its transpose $A^T$ has elements $A^T_{ij} = A_{ji}$
//! - **Row Echelon Form**: A matrix is in row echelon form when:
//!   - All rows consisting entirely of zeros are at the bottom
//!   - The leading coefficient (pivot) of each non-zero row is to the right of the pivot in the row
//!     above
//!   - All entries in a column below a pivot are zeros
//!
//! ## Storage Orientations
//!
//! This implementation supports two matrix storage orientations:
//!
//! - **Row major**: Elements are stored row by row (each row is contiguous in memory)
//! - **Column major**: Elements are stored column by column (each column is contiguous in memory)
//!
//! ## Examples
//!
//! ```
//! use harness_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{
//!     matrix::{DynamicDenseMatrix, RowMajor},
//!     vector::DynamicVector,
//!   },
//! };
//!
//! // Create a row-major matrix
//! let mut matrix = DynamicDenseMatrix::<f64, RowMajor>::new();
//!
//! // Add rows to the matrix
//! let row1 = DynamicVector::from([1.0, 2.0, 3.0]);
//! let row2 = DynamicVector::from([4.0, 5.0, 6.0]);
//! matrix.append_row(row1);
//! matrix.append_row(row2);
//!
//! // Access elements
//! let element = matrix.get_component(0, 1); // Gets element at row 0, column 1
//! assert_eq!(*element, 2.0);
//!
//! // Transform to row echelon form
//! let result = matrix.row_echelon_form();
//! ```

use std::{fmt::Debug, marker::PhantomData};

use super::{vector::DynamicVector, *};

/// Information about a pivot (non-zero entry) in a matrix.
///
/// When a matrix is transformed to row echelon form, pivots are the leading
/// non-zero entries in each row. This struct stores the row and column indices
/// of each pivot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PivotInfo {
  /// The row index of the pivot
  pub row: usize,
  /// The column index of the pivot
  pub col: usize,
}

/// Result of transforming a matrix to row echelon form.
///
/// This struct contains information about the rank of the matrix
/// and the positions of all pivots found during the transformation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEchelonOutput {
  /// The rank of the matrix (number of linearly independent rows/columns)
  pub rank:   usize,
  /// Positions of pivot elements in the row echelon form
  pub pivots: Vec<PivotInfo>,
}

// Sealed trait pattern to prevent external implementations of MatrixOrientation
mod sealed {
  pub trait Sealed {}
  impl Sealed for super::RowMajor {}
  impl Sealed for super::ColumnMajor {}
}

/// A marker trait for matrix storage orientation (RowMajor or ColumnMajor).
///
/// This trait is sealed, meaning only types defined in this crate can implement it.
/// The orientation determines how matrix elements are stored in memory, which affects
/// the performance characteristics of different operations.
pub trait MatrixOrientation: sealed::Sealed {}

/// Marker type for row-major matrix storage.
///
/// In row-major storage, elements of a row are contiguous in memory.
/// This is efficient for operations that access elements row by row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowMajor;
impl MatrixOrientation for RowMajor {}

/// Marker type for column-major matrix storage.
///
/// In column-major storage, elements of a column are contiguous in memory.
/// This is efficient for operations that access elements column by column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnMajor;
impl MatrixOrientation for ColumnMajor {}

/// A dynamically-sized matrix with elements from a field `F`.
///
/// ## Mathematical Representation
///
/// For a matrix $A \in F^{m \times n}$, the elements are represented as:
///
/// $$ A = \begin{pmatrix}
/// a_{11} & a_{12} & \cdots & a_{1n} \\
/// a_{21} & a_{22} & \cdots & a_{2n} \\
/// \vdots & \vdots & \ddots & \vdots \\
/// a_{m1} & a_{m2} & \cdots & a_{mn}
/// \end{pmatrix} $$
///
/// ## Storage Implementation
///
/// The storage orientation is determined by the type parameter `O`, which can be either
/// `RowMajor` or `ColumnMajor`. This affects the internal representation and the performance
/// characteristics of different operations:
///
/// - For `RowMajor`: Data is stored as a vector of row vectors
/// - For `ColumnMajor`: Data is stored as a vector of column vectors
///
/// ## Usage Notes
///
/// Operations that align with the storage orientation (e.g., row operations on a row-major matrix)
/// will generally be more efficient than those that don't.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicDenseMatrix<F, O: MatrixOrientation = RowMajor> {
  /// For [`RowMajor`]: `components` is a [`DynamicVector`] of rows (each row is a
  /// [`DynamicVector<F>`]).
  /// For [`ColumnMajor`]: `components` is a [`DynamicVector`] of columns (each col is a
  /// [`DynamicVector<F>`]).
  components:  DynamicVector<DynamicVector<F>>,
  /// The orientation of the matrix
  orientation: PhantomData<O>,
}

impl<F, O: MatrixOrientation> DynamicDenseMatrix<F, O> {
  /// Creates a new, empty `DynamicDenseMatrix` with the specified orientation.
  ///
  /// This constructor initializes a matrix with zero rows and zero columns.
  ///
  /// # Returns
  ///
  /// A new empty matrix with the orientation specified by the type parameter `O`.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::matrix::{ColumnMajor, DynamicDenseMatrix, RowMajor};
  ///
  /// // Create an empty row-major matrix of f64 values
  /// let row_major: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
  ///
  /// // Create an empty column-major matrix of f64 values
  /// let col_major: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
  /// ```
  pub const fn new() -> Self {
    Self { components: DynamicVector::new(Vec::new()), orientation: PhantomData }
  }
}

impl<F: Field + Copy> DynamicDenseMatrix<F, RowMajor> {
  /// Creates a new all zeros `DynamicDenseMatrix` with the specified number of rows and columns.
  ///
  /// # Arguments
  ///
  /// * `rows` - The number of rows in the matrix
  /// * `cols` - The number of columns in the matrix
  ///
  /// # Returns
  ///
  /// A new `DynamicDenseMatrix` with the specified number of rows and columns, all initialized to
  /// zero.
  pub fn zeros(rows: usize, cols: usize) -> Self {
    let mut mat = Self::new();
    for _ in 0..rows {
      mat.append_row(DynamicVector::zeros(cols));
    }
    mat
  }

  /// Returns the number of rows in the matrix.
  ///
  /// For a row-major matrix, this is the number of row vectors stored.
  ///
  /// # Returns
  ///
  /// The number of rows in the matrix
  pub const fn num_rows(&self) -> usize {
    self.components.dimension() // Outer vector stores rows
  }

  /// Returns the number of columns in the matrix.
  ///
  /// For a row-major matrix, this is the length of the first row vector (if any).
  /// Assumes a non-ragged matrix if rows > 0.
  ///
  /// # Returns
  ///
  /// The number of columns in the matrix
  pub fn num_cols(&self) -> usize {
    if self.components.dimension() == 0 {
      0
    } else {
      self.components.components()[0].dimension()
    }
  }

  /// Appends a new column to the matrix.
  ///
  /// If the matrix is empty, the column's elements become singleton rows.
  /// Otherwise, each element of the column is appended to the corresponding row.
  ///
  /// # Arguments
  ///
  /// * `column` - The column vector to append
  ///
  /// # Panics
  ///
  /// Panics if the column's length doesn't match the number of existing rows (when the matrix is
  /// not empty).
  ///
  /// # Warning
  ///
  /// For a row-major matrix, this operation requires updating every row vector.
  /// If you're building a matrix primarily by adding columns, consider using
  /// a column-major matrix for better performance.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::{
  ///   matrix::{DynamicDenseMatrix, RowMajor},
  ///   vector::DynamicVector,
  /// };
  ///
  /// let mut matrix = DynamicDenseMatrix::<f64, RowMajor>::new();
  ///
  /// // Add a column to the empty matrix (will create rows)
  /// let col = DynamicVector::from([1.0, 2.0]);
  /// matrix.append_column(&col);
  ///
  /// // Add another column
  /// let col2 = DynamicVector::from([3.0, 4.0]);
  /// matrix.append_column(&col2);
  /// ```
  pub fn append_column(&mut self, column: &DynamicVector<F>) {
    let num_r = self.num_rows();
    if num_r == 0 {
      if column.dimension() == 0 {
        return;
      }
      for i in 0..column.dimension() {
        self.components.components_mut().push(DynamicVector::new(vec![*column.get_component(i)]));
      }
    } else {
      assert_eq!(num_r, column.dimension(), "Column length must match the number of rows");
      for i in 0..num_r {
        self.components.components_mut()[i].append(*column.get_component(i));
      }
    }
  }

  /// Returns a new DynamicVector representing the column at the given index.
  ///
  /// For a row-major matrix, this requires extracting the element at position `index`
  /// from each row vector.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the column to retrieve (0-based)
  ///
  /// # Returns
  ///
  /// A new DynamicVector containing the elements of the specified column
  ///
  /// # Panics
  ///
  /// Panics if the column index is out of bounds.
  ///
  /// # Warning
  ///
  /// For a row-major matrix, this is a more expensive operation as it requires
  /// reading from each row vector. If you need to access columns frequently,
  /// consider using a column-major matrix instead.
  pub fn get_column(&self, index: usize) -> DynamicVector<F> {
    let num_r = self.num_rows();
    if num_r == 0 {
      return DynamicVector::new(Vec::new());
    }
    assert!(index < self.num_cols(), "Column index out of bounds");
    let mut col_components = Vec::with_capacity(num_r);
    for i in 0..num_r {
      col_components.push(*self.components.components()[i].get_component(index));
    }
    DynamicVector::new(col_components)
  }

  /// Sets the column at the given index with the provided DynamicVector.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the column to set (0-based)
  /// * `column` - The new column vector
  ///
  /// # Panics
  ///
  /// - Panics if the column index is out of bounds
  /// - Panics if the column's length doesn't match the number of rows
  ///
  /// # Warning
  ///
  /// For a row-major matrix, this is a more expensive operation as it requires
  /// updating every row vector. If you need to modify columns frequently,
  /// consider using a column-major matrix instead.
  pub fn set_column(&mut self, index: usize, column: &DynamicVector<F>) {
    let num_r = self.num_rows();
    assert_eq!(num_r, column.dimension(), "New column length must match the number of rows");
    if num_r == 0 {
      return;
    }
    assert!(index < self.num_cols(), "Column index out of bounds");
    for i in 0..num_r {
      self.components.components_mut()[i].set_component(index, *column.get_component(i));
    }
  }

  /// Appends a new row to the matrix.
  ///
  /// # Arguments
  ///
  /// * `row` - The row vector to append
  ///
  /// # Panics
  ///
  /// Panics if the row's length doesn't match the number of existing columns (when the matrix is
  /// not empty).
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::{
  ///   matrix::{DynamicDenseMatrix, RowMajor},
  ///   vector::DynamicVector,
  /// };
  ///
  /// let mut matrix = DynamicDenseMatrix::<f64, RowMajor>::new();
  ///
  /// // Add rows to the matrix
  /// let row1 = DynamicVector::from([1.0, 2.0, 3.0]);
  /// let row2 = DynamicVector::from([4.0, 5.0, 6.0]);
  /// matrix.append_row(row1);
  /// matrix.append_row(row2);
  /// ```
  pub fn append_row(&mut self, row: DynamicVector<F>) {
    if self.num_rows() > 0 {
      assert_eq!(
        self.num_cols(),
        row.dimension(),
        "New row length must match existing number of columns"
      );
    }
    self.components.components_mut().push(row);
  }

  /// Returns a reference to the row at the given index.
  ///
  /// For a row-major matrix, this directly accesses the stored row vector.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the row to retrieve (0-based)
  ///
  /// # Returns
  ///
  /// A reference to the row vector at the specified index
  ///
  /// # Panics
  ///
  /// Panics if the row index is out of bounds.
  pub fn get_row(&self, index: usize) -> &DynamicVector<F> {
    assert!(index < self.num_rows(), "Row index out of bounds");
    &self.components.components()[index]
  }

  /// Sets the row at the given index with the provided DynamicVector.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the row to set (0-based)
  /// * `row` - The new row vector
  ///
  /// # Panics
  ///
  /// - Panics if the row index is out of bounds
  /// - Panics if the row's length doesn't match the number of columns
  pub fn set_row(&mut self, index: usize, row: DynamicVector<F>) {
    assert!(index < self.num_rows(), "Row index out of bounds");
    if self.num_rows() > 0 {
      assert_eq!(
        self.num_cols(),
        row.dimension(),
        "New row length must match existing number of columns"
      );
    }
    self.components.components_mut()[index] = row;
  }

  /// Returns the component at the given row and column.
  ///
  /// # Arguments
  ///
  /// * `row` - The row index (0-based)
  /// * `col` - The column index (0-based)
  ///
  /// # Returns
  ///
  /// A reference to the component at the specified position
  ///
  /// # Panics
  ///
  /// Panics if either the row or column index is out of bounds.
  pub fn get_component(&self, row: usize, col: usize) -> &F {
    assert!(row < self.num_rows(), "Row index out of bounds");
    assert!(col < self.num_cols(), "Column index out of bounds");
    self.components.components()[row].get_component(col)
  }

  /// Sets the component at the given row and column to the given value.
  ///
  /// # Arguments
  ///
  /// * `row` - The row index (0-based)
  /// * `col` - The column index (0-based)
  /// * `value` - The value to set at the specified position
  ///
  /// # Panics
  ///
  /// Panics if either the row or column index is out of bounds.
  pub fn set_component(&mut self, row: usize, col: usize, value: F) {
    assert!(row < self.num_rows(), "Row index out of bounds");
    assert!(col < self.num_cols(), "Column index out of bounds");
    self.components.components_mut()[row].set_component(col, value);
  }

  /// Converts this row-major matrix to a column-major matrix by transposing it.
  ///
  /// The transpose of a matrix $A$ is denoted $A^T$ and has entries $A^T_{ij} = A_{ji}$.
  ///
  /// # Returns
  ///
  /// A new column-major matrix that is the transpose of this matrix
  pub fn transpose(self) -> DynamicDenseMatrix<F, ColumnMajor> {
    DynamicDenseMatrix { components: self.components, orientation: PhantomData }
  }

  /// Transforms this matrix into row echelon form using Gaussian elimination.
  ///
  /// Row echelon form has the following properties:
  /// - All rows consisting entirely of zeros are at the bottom
  /// - The leading coefficient (pivot) of each non-zero row is to the right of the pivot in the row
  ///   above
  /// - All entries in a column below a pivot are zeros
  ///
  /// This method performs in-place transformation of the matrix.
  ///
  /// # Returns
  ///
  /// A `RowEchelonOutput` containing the rank of the matrix and the positions of all pivots
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::{
  ///   matrix::{DynamicDenseMatrix, RowMajor},
  ///   vector::DynamicVector,
  /// };
  ///
  /// let mut matrix = DynamicDenseMatrix::<f64, RowMajor>::new();
  /// // Add some rows
  /// matrix.append_row(DynamicVector::from([1.0, 2.0, 3.0]));
  /// matrix.append_row(DynamicVector::from([4.0, 5.0, 6.0]));
  /// matrix.append_row(DynamicVector::from([7.0, 8.0, 9.0]));
  ///
  /// // Transform to row echelon form
  /// let result = matrix.row_echelon_form();
  ///
  /// // The result contains the rank and pivot positions
  /// assert_eq!(result.rank, 2); // The matrix has rank 2
  /// ```
  pub fn row_echelon_form(&mut self) -> RowEchelonOutput {
    let matrix = self.components.components_mut();
    if matrix.is_empty() || matrix[0].dimension() == 0 {
      return RowEchelonOutput { rank: 0, pivots: Vec::new() };
    }
    let rows = matrix.len();
    let cols = matrix[0].dimension();
    let mut lead = 0; // current pivot column
    let mut rank = 0;
    let mut pivots = Vec::new();

    for r in 0..rows {
      if lead >= cols {
        break;
      }
      let mut i = r;
      while matrix[i].get_component(lead).is_zero() {
        i += 1;
        if i == rows {
          i = r;
          lead += 1;
          if lead == cols {
            return RowEchelonOutput { rank, pivots };
          }
        }
      }
      matrix.swap(i, r);

      pivots.push(PivotInfo { row: r, col: lead });

      let pivot_val = *matrix[r].get_component(lead);
      let inv_pivot = pivot_val.multiplicative_inverse();

      for j in lead..cols {
        let val = *matrix[r].get_component(j);
        matrix[r].set_component(j, val * inv_pivot);
      }

      for i_row in 0..rows {
        if i_row != r {
          let factor = *matrix[i_row].get_component(lead);
          if !factor.is_zero() {
            for j_col in lead..cols {
              let val_r_j_col = *matrix[r].get_component(j_col);
              let term = factor * val_r_j_col;
              let val_i_row_j_col = *matrix[i_row].get_component(j_col); // Read from current row (i_row)
              matrix[i_row].set_component(j_col, val_i_row_j_col - term);
            }
          }
        }
      }
      lead += 1;
      rank += 1;
    }
    RowEchelonOutput { rank, pivots }
  }

  /// Computes a basis for the image (column space) of the matrix.
  /// The image is the span of the columns of the matrix.
  /// This method finds the pivot columns by transforming a copy of the matrix to RREF.
  /// The corresponding columns from the *original* matrix form the basis.
  /// This method does not modify `self`.
  pub fn image(&self) -> Vec<DynamicVector<F>> {
    if self.num_rows() == 0 || self.num_cols() == 0 {
      return Vec::new();
    }

    let mut rref_candidate = self.clone();
    let echelon_output = rref_candidate.row_echelon_form(); // Modifies rref_candidate

    let mut pivot_col_indices: Vec<usize> = echelon_output.pivots.iter().map(|p| p.col).collect();
    pivot_col_indices.sort_unstable();
    pivot_col_indices.dedup();

    let mut image_basis: Vec<DynamicVector<F>> = Vec::new();
    for &col_idx in &pivot_col_indices {
      image_basis.push(self.get_column(col_idx)); // get_column for RowMajor returns owned
                                                  // DynamicVector
    }
    image_basis
  }

  /// Computes a basis for the kernel (null space) of the matrix.
  /// The kernel is the set of all vectors x such that Ax = 0.
  /// This method returns a vector of `DynamicVector<F>` representing the basis vectors for the
  /// kernel. An empty vector is returned if the kernel is the zero space (e.g., for an invertible
  /// matrix, or if num_cols is 0). This method does not modify `self`.
  pub fn kernel(&self) -> Vec<DynamicVector<F>> {
    if self.num_cols() == 0 {
      return Vec::new();
    }

    if self.num_rows() == 0 {
      let mut basis: Vec<DynamicVector<F>> = Vec::with_capacity(self.num_cols());
      for i in 0..self.num_cols() {
        let mut v_data = vec![F::zero(); self.num_cols()];
        if i < v_data.len() {
          v_data[i] = F::one();
        }
        basis.push(DynamicVector::new(v_data));
      }
      return basis;
    }

    let mut rref_matrix = self.clone();
    let echelon_output = rref_matrix.row_echelon_form();

    let num_cols = rref_matrix.num_cols();
    let num_rows_of_rref = rref_matrix.num_rows();

    let mut is_pivot_col = vec![false; num_cols];
    for pivot_info in &echelon_output.pivots {
      if pivot_info.col < num_cols {
        is_pivot_col[pivot_info.col] = true;
      }
    }

    let mut free_col_indices: Vec<usize> = Vec::new();
    (0..num_cols).for_each(|j| {
      if !is_pivot_col[j] {
        free_col_indices.push(j);
      }
    });

    let mut kernel_basis: Vec<DynamicVector<F>> = Vec::new();

    for &free_idx in &free_col_indices {
      let mut basis_vector_comps = vec![F::zero(); num_cols];
      if free_idx < num_cols {
        basis_vector_comps[free_idx] = F::one();
      }

      for pivot_info in &echelon_output.pivots {
        let pivot_col = pivot_info.col;
        let pivot_row = pivot_info.row;

        if pivot_col < num_cols && free_idx < num_cols && pivot_row < num_rows_of_rref {
          let coefficient = *rref_matrix.get_component(pivot_row, free_idx);
          basis_vector_comps[pivot_col] = -coefficient;
        }
      }
      kernel_basis.push(DynamicVector::new(basis_vector_comps));
    }
    kernel_basis
  }
}

impl<F: Field + Copy> DynamicDenseMatrix<F, ColumnMajor> {
  /// Creates a new all zeros `DynamicDenseMatrix` with the specified number of rows and columns.
  ///
  /// # Arguments
  ///
  /// * `rows` - The number of rows in the matrix
  /// * `cols` - The number of columns in the matrix
  ///
  /// # Returns
  ///
  /// A new `DynamicDenseMatrix` with the specified number of rows and columns, all initialized to
  /// zero.
  pub fn zeros(rows: usize, cols: usize) -> Self {
    let mut mat = Self::new();
    for _ in 0..cols {
      mat.append_column(DynamicVector::zeros(rows));
    }
    mat
  }

  /// Returns the number of rows in the matrix.
  ///
  /// For a column-major matrix, this is the dimension of the first column vector (if any).
  ///
  /// # Returns
  ///
  /// The number of rows in the matrix
  pub fn num_rows(&self) -> usize {
    if self.components.dimension() == 0 {
      0
    } else {
      self.components.components()[0].dimension()
    }
  }

  /// Returns the number of columns in the matrix.
  ///
  /// For a column-major matrix, this is the number of column vectors stored.
  ///
  /// # Returns
  ///
  /// The number of columns in the matrix
  pub const fn num_cols(&self) -> usize { self.components.dimension() }

  /// Appends a new column to the matrix.
  ///
  /// For a column-major matrix, this is an efficient operation since columns are stored directly.
  ///
  /// # Arguments
  ///
  /// * `column` - The column vector to append
  ///
  /// # Panics
  ///
  /// Panics if the column's length doesn't match the number of existing rows (when the matrix is
  /// not empty).
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::{
  ///   matrix::{ColumnMajor, DynamicDenseMatrix},
  ///   vector::DynamicVector,
  /// };
  ///
  /// let mut matrix = DynamicDenseMatrix::<f64, ColumnMajor>::new();
  ///
  /// // Add columns to the matrix
  /// let col1 = DynamicVector::from([1.0, 2.0, 3.0]);
  /// let col2 = DynamicVector::from([4.0, 5.0, 6.0]);
  /// matrix.append_column(col1);
  /// matrix.append_column(col2);
  /// ```
  pub fn append_column(&mut self, column: DynamicVector<F>) {
    if self.num_cols() > 0 {
      assert_eq!(
        self.num_rows(),
        column.dimension(),
        "New column length must match existing number of rows"
      );
    }
    self.components.components_mut().push(column); // Add the new column vector
  }

  /// Returns a reference to the column at the given index.
  ///
  /// For a column-major matrix, this is an efficient operation since columns are stored directly.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the column to retrieve (0-based)
  ///
  /// # Returns
  ///
  /// A reference to the column vector at the specified index
  ///
  /// # Panics
  ///
  /// Panics if the column index is out of bounds.
  pub fn get_column(&self, index: usize) -> &DynamicVector<F> {
    assert!(index < self.num_cols(), "Column index out of bounds");
    &self.components.components()[index]
  }

  /// Sets the column at the given index with the provided DynamicVector.
  ///
  /// For a column-major matrix, this is an efficient operation since columns are stored directly.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the column to set (0-based)
  /// * `column` - The new column vector
  ///
  /// # Panics
  ///
  /// - Panics if the column index is out of bounds
  /// - Panics if the column's length doesn't match the number of rows
  pub fn set_column(&mut self, index: usize, column: DynamicVector<F>) {
    assert!(index < self.num_cols(), "Column index out of bounds");
    if self.num_cols() > 0 {
      assert_eq!(
        self.num_rows(),
        column.dimension(),
        "New column length must match existing number of rows"
      );
    }
    self.components.components_mut()[index] = column;
  }

  /// Appends a new row to the matrix.
  ///
  /// # Arguments
  ///
  /// * `row` - The row vector to append
  ///
  /// # Panics
  ///
  /// Panics if the row's length doesn't match the number of existing columns.
  ///
  /// # Warning
  ///
  /// For a column-major matrix, this is a more expensive operation as it requires
  /// updating every column vector. If you need to add many rows, consider using
  /// a row-major matrix or building the matrix from columns instead.
  pub fn append_row(&mut self, row: &DynamicVector<F>) {
    let num_c = self.num_cols();
    if num_c == 0 {
      if row.dimension() == 0 {
        return;
      }
      for i in 0..row.dimension() {
        self.components.components_mut().push(DynamicVector::new(vec![*row.get_component(i)]));
      }
    } else {
      assert_eq!(num_c, row.dimension(), "Row length must match the number of columns");
      for i in 0..num_c {
        self.components.components_mut()[i].append(*row.get_component(i));
      }
    }
  }

  /// Returns a new DynamicVector representing the row at the given index.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the row to retrieve (0-based)
  ///
  /// # Returns
  ///
  /// A new DynamicVector containing the elements of the specified row
  ///
  /// # Panics
  ///
  /// Panics if the row index is out of bounds.
  ///
  /// # Warning
  ///
  /// For a column-major matrix, this is a more expensive operation as it requires
  /// reading from each column vector. If you need to access rows frequently,
  /// consider using a row-major matrix instead.
  pub fn get_row(&self, index: usize) -> DynamicVector<F> {
    let num_c = self.num_cols();
    if num_c == 0 {
      return DynamicVector::new(Vec::new());
    }
    assert!(index < self.num_rows(), "Row index out of bounds");
    let mut row_components = Vec::with_capacity(num_c);
    for i in 0..num_c {
      row_components.push(*self.components.components()[i].get_component(index));
    }
    DynamicVector::new(row_components)
  }

  /// Sets the row at the given index with the provided DynamicVector.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the row to set (0-based)
  /// * `row` - The new row vector
  ///
  /// # Panics
  ///
  /// - Panics if the row index is out of bounds
  /// - Panics if the row's length doesn't match the number of columns
  ///
  /// # Warning
  ///
  /// For a column-major matrix, this is a more expensive operation as it requires
  /// updating every column vector. If you need to modify rows frequently,
  /// consider using a row-major matrix instead.
  pub fn set_row(&mut self, index: usize, row: &DynamicVector<F>) {
    let num_c = self.num_cols();
    assert_eq!(num_c, row.dimension(), "New row length must match the number of columns");
    if num_c == 0 {
      return; // If no columns, setting an empty row to an empty matrix is fine.
    }
    assert!(index < self.num_rows(), "Row index out of bounds");

    for i in 0..num_c {
      self.components.components_mut()[i].set_component(index, *row.get_component(i));
    }
  }

  /// Returns the component at the given row and column.
  ///
  /// # Arguments
  ///
  /// * `row` - The row index (0-based)
  /// * `col` - The column index (0-based)
  ///
  /// # Returns
  ///
  /// A reference to the component at the specified position
  ///
  /// # Panics
  ///
  /// Panics if either the row or column index is out of bounds.
  pub fn get_component(&self, row: usize, col: usize) -> &F {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds");
    self.components.components()[col].get_component(row)
  }

  /// Sets the component at the given row and column to the given value.
  ///
  /// # Arguments
  ///
  /// * `row` - The row index (0-based)
  /// * `col` - The column index (0-based)
  /// * `value` - The value to set at the specified position
  ///
  /// # Panics
  ///
  /// Panics if either the row or column index is out of bounds.
  pub fn set_component(&mut self, row: usize, col: usize, value: F) {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds");
    self.components.components_mut()[col].set_component(row, value);
  }

  /// Converts this column-major matrix to a row-major matrix by transposing it.
  ///
  /// The transpose of a matrix $A$ is denoted $A^T$ and has entries $A^T_{ij} = A_{ji}$.
  ///
  /// # Returns
  ///
  /// A new row-major matrix that is the transpose of this matrix
  pub fn transpose(self) -> DynamicDenseMatrix<F, RowMajor> {
    DynamicDenseMatrix { components: self.components, orientation: PhantomData }
  }

  /// Transforms this matrix into row echelon form using Gaussian elimination.
  ///
  /// Row echelon form has the following properties:
  /// - All rows consisting entirely of zeros are at the bottom
  /// - The leading coefficient (pivot) of each non-zero row is to the right of the pivot in the row
  ///   above
  /// - All entries in a column below a pivot are zeros
  ///
  /// This method performs in-place transformation of the matrix.
  ///
  /// # Returns
  ///
  /// A `RowEchelonOutput` containing the rank of the matrix and the positions of all pivots
  ///
  /// # Warning
  ///
  /// While the algorithm for row echelon form works for column-major matrices,
  /// it may be less efficient than for row-major matrices since row operations
  /// require accessing multiple column vectors.
  pub fn row_echelon_form(&mut self) -> RowEchelonOutput {
    let matrix = self.components.components_mut();
    if matrix.is_empty() || matrix[0].dimension() == 0 {
      return RowEchelonOutput { rank: 0, pivots: Vec::new() };
    }
    let num_actual_cols = matrix.len();
    let num_actual_rows = matrix[0].dimension();

    let mut pivot_row_idx = 0;
    let mut rank = 0;
    let mut pivots = Vec::new();

    for pivot_col_idx in 0..num_actual_cols {
      if pivot_row_idx >= num_actual_rows {
        break;
      }

      let mut i_search_row = pivot_row_idx;
      while i_search_row < num_actual_rows
        && matrix[pivot_col_idx].get_component(i_search_row).is_zero()
      {
        i_search_row += 1;
      }

      if i_search_row < num_actual_rows {
        // Found a pivot in this column
        if i_search_row != pivot_row_idx {
          // Swap rows to bring pivot to pivot_row_idx
          (0..num_actual_cols).for_each(|k_col| {
            // Iterate through all columns to swap elements
            let temp = *matrix[k_col].get_component(i_search_row);
            let val_at_pivot_row = *matrix[k_col].get_component(pivot_row_idx);
            matrix[k_col].set_component(i_search_row, val_at_pivot_row);
            matrix[k_col].set_component(pivot_row_idx, temp);
          });
        }

        pivots.push(PivotInfo { row: pivot_row_idx, col: pivot_col_idx });

        let pivot_val = *matrix[pivot_col_idx].get_component(pivot_row_idx);

        if !pivot_val.is_zero() {
          let inv_pivot_val = pivot_val.multiplicative_inverse();
          (pivot_col_idx..num_actual_cols).for_each(|k_col| {
            let current_val = *matrix[k_col].get_component(pivot_row_idx);
            matrix[k_col].set_component(pivot_row_idx, current_val * inv_pivot_val);
          });
        }

        for k_row in 0..num_actual_rows {
          if k_row != pivot_row_idx {
            let factor = *matrix[pivot_col_idx].get_component(k_row);
            if !factor.is_zero() {
              (pivot_col_idx..num_actual_cols).for_each(|j_col_elim| {
                let val_from_pivot_row_at_j_col = *matrix[j_col_elim].get_component(pivot_row_idx);
                let term_to_subtract = factor * val_from_pivot_row_at_j_col;
                let current_val_in_k_row_at_j_col = *matrix[j_col_elim].get_component(k_row);
                matrix[j_col_elim]
                  .set_component(k_row, current_val_in_k_row_at_j_col - term_to_subtract);
              });
            }
          }
        }
        pivot_row_idx += 1;
        rank += 1;
      }
    }
    RowEchelonOutput { rank, pivots }
  }

  /// Computes a basis for the image (column space) of the matrix.
  /// The image is the span of the columns of the matrix.
  /// This method finds the pivot columns by transforming a copy of the matrix to RREF.
  /// The corresponding columns from the *original* matrix form the basis.
  /// This method does not modify `self`.
  pub fn image(&self) -> Vec<DynamicVector<F>> {
    if self.num_rows() == 0 || self.num_cols() == 0 {
      return Vec::new();
    }

    let mut rref_candidate = self.clone();
    let echelon_output = rref_candidate.row_echelon_form(); // Modifies rref_candidate

    let mut pivot_col_indices: Vec<usize> = echelon_output.pivots.iter().map(|p| p.col).collect();
    pivot_col_indices.sort_unstable();
    pivot_col_indices.dedup();

    let mut image_basis: Vec<DynamicVector<F>> = Vec::new();
    for &col_idx in &pivot_col_indices {
      // get_column for ColumnMajor returns &DynamicVector, so clone is needed.
      image_basis.push(self.get_column(col_idx).clone());
    }
    image_basis
  }

  /// Computes a basis for the kernel (null space) of the matrix.
  /// The kernel is the set of all vectors x such that Ax = 0.
  /// This method returns a vector of `DynamicVector<F>` representing the basis vectors for the
  /// kernel. An empty vector is returned if the kernel is the zero space (e.g., for an invertible
  /// matrix, or if num_cols is 0). This method does not modify `self`.
  pub fn kernel(&self) -> Vec<DynamicVector<F>> {
    if self.num_cols() == 0 {
      return Vec::new();
    }

    if self.num_rows() == 0 {
      let mut basis: Vec<DynamicVector<F>> = Vec::with_capacity(self.num_cols());
      for i in 0..self.num_cols() {
        let mut v_data = vec![F::zero(); self.num_cols()];
        if i < v_data.len() {
          v_data[i] = F::one();
        }
        basis.push(DynamicVector::new(v_data));
      }
      return basis;
    }

    let mut rref_matrix = self.clone();
    let echelon_output = rref_matrix.row_echelon_form();

    let num_cols = rref_matrix.num_cols();
    let num_rows_of_rref = rref_matrix.num_rows();

    let mut is_pivot_col = vec![false; num_cols];
    for pivot_info in &echelon_output.pivots {
      if pivot_info.col < num_cols {
        is_pivot_col[pivot_info.col] = true;
      }
    }

    let mut free_col_indices: Vec<usize> = Vec::new();
    (0..num_cols).for_each(|j| {
      if !is_pivot_col[j] {
        free_col_indices.push(j);
      }
    });

    let mut kernel_basis: Vec<DynamicVector<F>> = Vec::new();

    for &free_idx in &free_col_indices {
      let mut basis_vector_comps = vec![F::zero(); num_cols];
      if free_idx < num_cols {
        basis_vector_comps[free_idx] = F::one();
      }

      for pivot_info in &echelon_output.pivots {
        let pivot_col = pivot_info.col;
        let pivot_row = pivot_info.row;

        if pivot_col < num_cols && free_idx < num_cols && pivot_row < num_rows_of_rref {
          let coefficient = *rref_matrix.get_component(pivot_row, free_idx);
          basis_vector_comps[pivot_col] = -coefficient;
        }
      }
      kernel_basis.push(DynamicVector::new(basis_vector_comps));
    }
    kernel_basis
  }
}

impl<T: Field + Copy> Mul<DynamicVector<T>> for DynamicDenseMatrix<T, RowMajor> {
  type Output = DynamicVector<T>;

  fn mul(self, rhs: DynamicVector<T>) -> Self::Output {
    assert_eq!(self.num_cols(), rhs.dimension(), "Matrix-vector dimension mismatch");

    let mut result = vec![T::zero(); self.num_rows()];
    (0..self.num_rows()).for_each(|i| {
      for j in 0..self.num_cols() {
        result[i] += *self.get_component(i, j) * *rhs.get_component(j);
      }
    });

    DynamicVector::new(result)
  }
}

impl<T: Field + Copy> Mul<DynamicVector<T>> for DynamicDenseMatrix<T, ColumnMajor> {
  type Output = DynamicVector<T>;

  fn mul(self, rhs: DynamicVector<T>) -> Self::Output {
    assert_eq!(self.num_cols(), rhs.dimension(), "Matrix-vector dimension mismatch");

    let mut result = vec![T::zero(); self.num_rows()];
    (0..self.num_rows()).for_each(|i| {
      for j in 0..self.num_cols() {
        result[i] += *self.get_component(i, j) * *rhs.get_component(j);
      }
    });

    DynamicVector::new(result)
  }
}

impl<T: Field + Copy> Mul<Self> for DynamicDenseMatrix<T, RowMajor> {
  type Output = Self;

  fn mul(self, rhs: Self) -> Self::Output {
    let mut result = Self::new();
    for i in 0..self.num_rows() {
      let mut new_row = DynamicVector::<T>::zeros(rhs.num_cols());
      for j in 0..rhs.num_cols() {
        let col = rhs.get_column(j);
        let mut sum = T::zero();
        for k in 0..self.num_cols() {
          sum += *self.get_component(i, k) * *col.get_component(k);
        }
        new_row.set_component(j, sum);
      }
      result.append_row(new_row);
    }
    result
  }
}

impl<T: Field + Copy> Mul<Self> for DynamicDenseMatrix<T, ColumnMajor> {
  type Output = Self;

  fn mul(self, rhs: Self) -> Self::Output {
    assert_eq!(
      self.num_cols(),
      rhs.num_rows(),
      "Matrix dimensions incompatible for multiplication"
    );
    let m = self.num_rows();
    let n = self.num_cols(); // common dimension, also rhs.num_rows()
    let p = rhs.num_cols();

    let mut result_matrix = Self::new();

    for j_res in 0..p {
      // For each column j_res of the result matrix C
      let mut new_col_components = Vec::with_capacity(m);
      for i_res in 0..m {
        // For each row i_res in that result column
        let mut sum = T::zero();
        for k in 0..n {
          // Summation index
          // C(i_res, j_res) = sum_k A(i_res, k) * B(k, j_res)
          // self is A (ColumnMajor), rhs is B (ColumnMajor)
          sum += *self.get_component(i_res, k) * *rhs.get_component(k, j_res);
        }
        new_col_components.push(sum);
      }
      result_matrix.append_column(DynamicVector::new(new_col_components));
    }
    result_matrix
  }
}

impl<T: Field + Copy> Mul<DynamicDenseMatrix<T, RowMajor>> for DynamicDenseMatrix<T, ColumnMajor> {
  type Output = Self;

  fn mul(self, rhs: DynamicDenseMatrix<T, RowMajor>) -> Self::Output {
    assert_eq!(
      self.num_cols(),
      rhs.num_rows(),
      "Matrix dimensions incompatible for multiplication"
    );
    let m = self.num_rows();
    let n = self.num_cols(); // common dimension, also rhs.num_rows()
    let p = rhs.num_cols();

    let mut result_matrix = Self::new();

    for j_res in 0..p {
      // For each column j_res of the result matrix C
      let mut new_col_components = Vec::with_capacity(m);
      for i_res in 0..m {
        // For each row i_res in that result column
        let mut sum = T::zero();
        for k in 0..n {
          // Summation index
          // C(i_res, j_res) = sum_k A(i_res, k) * B(k, j_res)
          // self is A (RowMajor), rhs is B (ColumnMajor)
          sum += *self.get_component(i_res, k) * *rhs.get_component(k, j_res);
        }
        new_col_components.push(sum);
      }
      result_matrix.append_column(DynamicVector::new(new_col_components));
    }
    result_matrix
  }
}

#[cfg(test)]
mod tests {
  #![allow(non_snake_case)]
  use super::*;
  use crate::{algebras::boolean::Boolean, fixtures::Mod7};

  // Test constructor and basic properties
  #[test]
  fn test_new_matrix_properties() {
    // RowMajor f64
    let m_rm_f64: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    assert_eq!(m_rm_f64.num_rows(), 0);
    assert_eq!(m_rm_f64.num_cols(), 0);

    // ColumnMajor Boolean
    let m_cm_bool: DynamicDenseMatrix<Boolean, ColumnMajor> = DynamicDenseMatrix::new();
    assert_eq!(m_cm_bool.num_rows(), 0);
    assert_eq!(m_cm_bool.num_cols(), 0);
  }

  // Test append_row, get_row, set_row for RowMajor
  #[test]
  fn test_row_operations_row_major_mod7() {
    let mut m: DynamicDenseMatrix<Mod7, RowMajor> = DynamicDenseMatrix::new();
    let r0_data = vec![Mod7::new(1), Mod7::new(2)];
    let r1_data = vec![Mod7::new(3), Mod7::new(4)];
    let r0 = DynamicVector::new(r0_data.clone());
    let r1 = DynamicVector::new(r1_data.clone());

    m.append_row(r0);
    assert_eq!(m.num_rows(), 1);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(0).components(), &r0_data);

    m.append_row(r1);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(1).components(), &r1_data);

    let r_new_data = vec![Mod7::new(5), Mod7::new(6)];
    let r_new = DynamicVector::new(r_new_data.clone());
    m.set_row(0, r_new);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(0).components(), &r_new_data);
    assert_eq!(m.get_row(1).components(), &r1_data);
  }

  // Test append_column, get_column, set_column for RowMajor
  #[test]
  fn test_column_operations_row_major_f64() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();

    // Append first column to empty matrix
    let c0_data = vec![1.0, 2.0];
    let c0 = DynamicVector::new(c0_data.clone());
    m.append_column(&c0);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 1);
    assert_eq!(m.get_column(0).components(), &c0_data);
    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(1, 0), 2.0);

    // Append second column
    let c1_data = vec![3.0, 4.0];
    let c1 = DynamicVector::new(c1_data.clone());
    m.append_column(&c1);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_column(1).components(), &c1_data);
    assert_eq!(*m.get_component(0, 1), 3.0);
    assert_eq!(*m.get_component(1, 1), 4.0);

    // Set a column
    let c_new_data = vec![5.0, 6.0];
    let c_new = DynamicVector::new(c_new_data.clone());
    m.set_column(0, &c_new);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_column(0).components(), &c_new_data);
    assert_eq!(m.get_column(1).components(), &c1_data);
  }

  // Test append_column, get_column, set_column for ColumnMajor
  #[test]
  fn test_column_operations_col_major_boolean() {
    let mut m: DynamicDenseMatrix<Boolean, ColumnMajor> = DynamicDenseMatrix::new();
    let c0_data = vec![Boolean(true), Boolean(false)];
    let c1_data = vec![Boolean(false), Boolean(true)];
    let c0 = DynamicVector::new(c0_data.clone());
    let c1 = DynamicVector::new(c1_data.clone());

    m.append_column(c0.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 1);
    assert_eq!(m.get_column(0).components(), &c0_data);

    m.append_column(c1.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_column(1).components(), &c1_data);

    let c_new_data = vec![Boolean(true), Boolean(true)];
    let c_new = DynamicVector::new(c_new_data.clone());
    m.set_column(0, c_new.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_column(0).components(), &c_new_data);
    assert_eq!(m.get_column(1).components(), &c1_data);
  }

  // Test append_row, get_row, set_row for ColumnMajor
  #[test]
  fn test_row_operations_col_major_f64() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();

    // Append first row to empty matrix
    let r0_data = vec![1.0, 2.0];
    let r0 = DynamicVector::new(r0_data.clone());
    m.append_row(&r0);
    assert_eq!(m.num_rows(), 1);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(0).components(), &r0_data);
    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(0, 1), 2.0);

    // Append second row
    let r1_data = vec![3.0, 4.0];
    let r1 = DynamicVector::new(r1_data.clone());
    m.append_row(&r1);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(1).components(), &r1_data);
    assert_eq!(*m.get_component(1, 0), 3.0);
    assert_eq!(*m.get_component(1, 1), 4.0);

    // Set a row
    let r_new_data = vec![5.0, 6.0];
    let r_new = DynamicVector::new(r_new_data.clone());
    m.set_row(0, &r_new);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(0).components(), &r_new_data);
    assert_eq!(m.get_row(1).components(), &r1_data);
  }

  // Test get_component, set_component for RowMajor and ColumnMajor
  #[test]
  fn test_get_set_component() {
    let mut m_rm: DynamicDenseMatrix<Mod7, RowMajor> = DynamicDenseMatrix::new();
    m_rm.append_row(DynamicVector::new(vec![Mod7::new(1), Mod7::new(2)]));
    m_rm.append_row(DynamicVector::new(vec![Mod7::new(3), Mod7::new(4)]));
    assert_eq!(*m_rm.get_component(0, 1), Mod7::new(2));
    m_rm.set_component(0, 1, Mod7::new(5));
    assert_eq!(*m_rm.get_component(0, 1), Mod7::new(5));

    let mut m_cm: DynamicDenseMatrix<Mod7, ColumnMajor> = DynamicDenseMatrix::new();
    m_cm.append_column(DynamicVector::new(vec![Mod7::new(1), Mod7::new(2)]));
    m_cm.append_column(DynamicVector::new(vec![Mod7::new(3), Mod7::new(4)]));
    assert_eq!(*m_cm.get_component(1, 0), Mod7::new(2));
    m_cm.set_component(1, 0, Mod7::new(6));
    assert_eq!(*m_cm.get_component(1, 0), Mod7::new(6));
  }

  // Test transpose
  #[test]
  fn test_transpose() {
    let mut m_rm: DynamicDenseMatrix<Mod7, RowMajor> = DynamicDenseMatrix::new();
    m_rm.append_row(DynamicVector::new(vec![Mod7::new(1), Mod7::new(2), Mod7::new(3)]));
    m_rm.append_row(DynamicVector::new(vec![Mod7::new(4), Mod7::new(5), Mod7::new(6)]));
    assert_eq!(m_rm.num_rows(), 2);
    assert_eq!(m_rm.num_cols(), 3);

    let m_cm: DynamicDenseMatrix<Mod7, ColumnMajor> = m_rm.transpose();
    assert_eq!(m_cm.num_rows(), 3);
    assert_eq!(m_cm.num_cols(), 2);

    assert_eq!(*m_cm.get_component(0, 0), Mod7::new(1));
    assert_eq!(*m_cm.get_component(1, 0), Mod7::new(2));
    assert_eq!(*m_cm.get_component(2, 0), Mod7::new(3));
    assert_eq!(*m_cm.get_component(0, 1), Mod7::new(4));
    assert_eq!(*m_cm.get_component(1, 1), Mod7::new(5));
    assert_eq!(*m_cm.get_component(2, 1), Mod7::new(6));

    // Transpose back
    let m_rm_again: DynamicDenseMatrix<Mod7, RowMajor> = m_cm.transpose();
    assert_eq!(m_rm_again.num_rows(), 2);
    assert_eq!(m_rm_again.num_cols(), 3);
    assert_eq!(*m_rm_again.get_component(0, 0), Mod7::new(1));
    assert_eq!(*m_rm_again.get_component(0, 1), Mod7::new(2));
    assert_eq!(*m_rm_again.get_component(0, 2), Mod7::new(3));
    assert_eq!(*m_rm_again.get_component(1, 0), Mod7::new(4));
    assert_eq!(*m_rm_again.get_component(1, 1), Mod7::new(5));
    assert_eq!(*m_rm_again.get_component(1, 2), Mod7::new(6));
  }

  #[test]
  #[should_panic]
  fn test_append_row_mismatch_cols_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::new(vec![1.0, 2.0]));
    m.append_row(DynamicVector::new(vec![3.0])); // Should panic
  }

  #[test]
  #[should_panic]
  fn test_append_column_mismatch_rows_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_column(&DynamicVector::new(vec![1.0, 2.0]));
    m.append_column(&DynamicVector::new(vec![3.0])); // Should panic
  }

  #[test]
  #[should_panic]
  fn test_set_row_mismatch_cols_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::new(vec![1.0, 2.0]));
    m.set_row(0, DynamicVector::new(vec![3.0])); // Should panic
  }

  #[test]
  #[should_panic]
  fn test_set_column_mismatch_rows_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_column(&DynamicVector::new(vec![1.0]));
    m.set_column(0, &DynamicVector::new(vec![3.0, 4.0, 5.0])); // Should panic
  }

  #[test]
  #[should_panic]
  fn test_append_column_mismatch_rows_col_major() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_column(DynamicVector::new(vec![1.0, 2.0]));
    m.append_column(DynamicVector::new(vec![3.0])); // Should panic
  }

  #[test]
  #[should_panic]
  fn test_append_row_mismatch_cols_col_major() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_row(&DynamicVector::new(vec![1.0, 2.0]));
    m.append_row(&DynamicVector::new(vec![3.0])); // Should panic
  }

  #[test]
  fn test_row_echelon_form_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::new(vec![1.0, 2.0, 3.0]));
    m.append_row(DynamicVector::new(vec![4.0, 5.0, 6.0]));
    m.append_row(DynamicVector::new(vec![7.0, 8.0, 9.0]));
    let result = m.row_echelon_form();
    assert_eq!(result.rank, 2);
    assert_eq!(result.pivots, vec![PivotInfo { row: 0, col: 0 }, PivotInfo { row: 1, col: 1 }]);

    assert_eq!(m.num_rows(), 3);
    assert_eq!(m.num_cols(), 3);

    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(0, 1), 0.0);
    assert_eq!(*m.get_component(0, 2), -1.0);
    assert_eq!(*m.get_component(1, 0), 0.0);
    assert_eq!(*m.get_component(1, 1), 1.0);
    assert_eq!(*m.get_component(1, 2), 2.0);
    assert_eq!(*m.get_component(2, 0), 0.0);
    assert_eq!(*m.get_component(2, 1), 0.0);
    assert_eq!(*m.get_component(2, 2), 0.0);
  }

  #[test]
  fn test_row_echelon_form_col_major() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_column(DynamicVector::new(vec![1.0, 2.0, 3.0]));
    m.append_column(DynamicVector::new(vec![4.0, 5.0, 6.0]));
    m.append_column(DynamicVector::new(vec![7.0, 8.0, 9.0]));
    let result = m.row_echelon_form();
    assert_eq!(result.rank, 2);
    assert_eq!(result.pivots, vec![PivotInfo { row: 0, col: 0 }, PivotInfo { row: 1, col: 1 }]);

    assert_eq!(m.num_rows(), 3);
    assert_eq!(m.num_cols(), 3);

    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(0, 1), 0.0);
    assert_eq!(*m.get_component(0, 2), -1.0);
    assert_eq!(*m.get_component(1, 0), 0.0);
    assert_eq!(*m.get_component(1, 1), 1.0);
    assert_eq!(*m.get_component(1, 2), 2.0);
    assert_eq!(*m.get_component(2, 0), 0.0);
    assert_eq!(*m.get_component(2, 1), 0.0);
    assert_eq!(*m.get_component(2, 2), 0.0);
  }

  // Helper function to check if a vector is in a list of vectors (basis)
  // This is a simple check, assumes vectors in basis are unique and non-zero for simplicity.
  // For more robust checks, one might need to check for linear independence and spanning.
  fn contains_vector<F: Field + Copy + PartialEq>(
    basis: &[DynamicVector<F>],
    vector: &DynamicVector<F>,
  ) -> bool {
    basis.iter().any(|v| v == vector)
  }

  #[test]
  fn test_image_kernel_row_major_simple() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    // A = [[1, 0, -1],
    //      [0, 1,  2]]
    m.append_row(DynamicVector::from(vec![1.0, 0.0, -1.0]));
    m.append_row(DynamicVector::from(vec![0.0, 1.0, 2.0]));

    let image = m.image();
    // Pivots are in col 0 and col 1. Image is span of original col 0 and col 1.
    let expected_image_basis = [
      DynamicVector::from(vec![1.0, 0.0]), // Original col 0
      DynamicVector::from(vec![0.0, 1.0]), // Original col 1
    ];
    assert_eq!(image.len(), 2);
    assert!(contains_vector(&image, &expected_image_basis[0]));
    assert!(contains_vector(&image, &expected_image_basis[1]));

    let kernel = m.kernel();
    // RREF is [[1,0,-1],[0,1,2]]. x1 - x3 = 0, x2 + 2x3 = 0.
    // x3 is free. x1 = x3, x2 = -2x3. Vector: [1, -2, 1]^T * x3
    let expected_kernel_basis = [DynamicVector::from(vec![1.0, -2.0, 1.0])];
    assert_eq!(kernel.len(), 1);
    assert!(contains_vector(&kernel, &expected_kernel_basis[0]));

    // Check Ax = 0 for kernel vectors
    for k_vec in &kernel {
      let mut Ax_components = vec![0.0; m.num_rows()];
      (0..m.num_rows()).for_each(|r| {
        let mut sum = 0.0;
        for c in 0..m.num_cols() {
          sum += m.get_component(r, c) * k_vec.get_component(c);
        }
        Ax_components[r] = sum;
      });
      let Ax = DynamicVector::new(Ax_components);
      let zero_vec = DynamicVector::new(vec![0.0; m.num_rows()]);
      assert_eq!(Ax, zero_vec, "Kernel vector validation failed: Ax != 0");
    }
  }

  #[test]
  fn test_image_kernel_col_major_simple() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    // A = [[1, 0],
    //      [0, 1],
    //      [-1, 2]]
    // This is the transpose of the RowMajor test for easier comparison logic.
    m.append_column(DynamicVector::from(vec![1.0, 0.0, -1.0]));
    m.append_column(DynamicVector::from(vec![0.0, 1.0, 2.0]));

    // For A (3x2), RREF would be [[1,0],[0,1],[0,0]]
    // Image basis: col 0, col 1 of original matrix
    let image = m.image();
    let expected_image_basis =
      [DynamicVector::from(vec![1.0, 0.0, -1.0]), DynamicVector::from(vec![0.0, 1.0, 2.0])];
    assert_eq!(image.len(), 2);
    assert!(contains_vector(&image, &expected_image_basis[0]));
    assert!(contains_vector(&image, &expected_image_basis[1]));

    // Kernel for this 3x2 matrix (rank 2) should be trivial (only zero vector)
    let kernel = m.kernel();
    assert_eq!(kernel.len(), 0, "Kernel should be trivial for this matrix");
  }

  #[test]
  fn test_image_kernel_identity_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::from(vec![1.0, 0.0]));
    m.append_row(DynamicVector::from(vec![0.0, 1.0]));

    let image = m.image();
    let expected_image_basis =
      [DynamicVector::from(vec![1.0, 0.0]), DynamicVector::from(vec![0.0, 1.0])];
    assert_eq!(image.len(), 2);
    assert!(contains_vector(&image, &expected_image_basis[0]));
    assert!(contains_vector(&image, &expected_image_basis[1]));

    let kernel = m.kernel();
    assert_eq!(kernel.len(), 0, "Kernel of identity matrix should be trivial");
  }

  #[test]
  fn test_image_kernel_zero_matrix_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::from(vec![0.0, 0.0]));
    m.append_row(DynamicVector::from(vec![0.0, 0.0]));

    let image = m.image();
    assert_eq!(image.len(), 0, "Image of zero matrix should be trivial");

    let kernel = m.kernel();
    // Kernel of 2x2 zero matrix is R^2, basis e.g., [[1,0],[0,1]]
    let expected_kernel_basis =
      [DynamicVector::from(vec![1.0, 0.0]), DynamicVector::from(vec![0.0, 1.0])];
    assert_eq!(kernel.len(), 2);
    // Order might differ, so check containment
    assert!(contains_vector(&kernel, &expected_kernel_basis[0]));
    assert!(contains_vector(&kernel, &expected_kernel_basis[1]));
  }

  #[test]
  fn test_image_kernel_dependent_cols_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    // A = [[1, 2, 3],
    //      [2, 4, 6]]
    // col2 = 2*col1, col3 = 3*col1. Rank = 1.
    m.append_row(DynamicVector::from(vec![1.0, 2.0, 3.0]));
    m.append_row(DynamicVector::from(vec![2.0, 4.0, 6.0]));

    let image = m.image();
    // RREF will have pivot in first col. Image is span of original first col.
    let expected_image_basis = [DynamicVector::from(vec![1.0, 2.0])];
    assert_eq!(image.len(), 1);
    assert!(contains_vector(&image, &expected_image_basis[0]));

    let kernel = m.kernel();
    // RREF: [[1, 2, 3], [0, 0, 0]]
    // x1 + 2x2 + 3x3 = 0. x2, x3 are free.
    // Basis vector 1 (x2=1, x3=0): [-2, 1, 0]^T
    // Basis vector 2 (x2=0, x3=1): [-3, 0, 1]^T
    let expected_kernel_vector1 = DynamicVector::from(vec![-2.0, 1.0, 0.0]);
    let expected_kernel_vector2 = DynamicVector::from(vec![-3.0, 0.0, 1.0]);
    assert_eq!(kernel.len(), 2);
    assert!(
      contains_vector(&kernel, &expected_kernel_vector1)
        || contains_vector(&kernel, &DynamicVector::from(vec![2.0, -1.0, 0.0]))
    );
    assert!(
      contains_vector(&kernel, &expected_kernel_vector2)
        || contains_vector(&kernel, &DynamicVector::from(vec![3.0, 0.0, -1.0]))
    );

    for k_vec in &kernel {
      let mut Ax_components = vec![0.0; m.num_rows()];
      (0..m.num_rows()).for_each(|r| {
        let mut sum = 0.0;
        for c in 0..m.num_cols() {
          sum += m.get_component(r, c) * k_vec.get_component(c);
        }
        Ax_components[r] = sum;
      });
      let Ax = DynamicVector::new(Ax_components);
      let zero_vec = DynamicVector::new(vec![0.0; m.num_rows()]);
      assert_eq!(Ax, zero_vec, "Kernel vector validation failed: Ax != 0");
    }
  }

  #[test]
  fn test_image_kernel_col_major_identity() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_column(DynamicVector::from(vec![1.0, 0.0]));
    m.append_column(DynamicVector::from(vec![0.0, 1.0]));

    let image = m.image();
    let expected_image_basis =
      [DynamicVector::from(vec![1.0, 0.0]), DynamicVector::from(vec![0.0, 1.0])];
    assert_eq!(image.len(), 2);
    assert!(contains_vector(&image, &expected_image_basis[0]));
    assert!(contains_vector(&image, &expected_image_basis[1]));

    let kernel = m.kernel();
    assert_eq!(kernel.len(), 0, "Kernel of identity matrix should be trivial");
  }

  #[test]
  fn test_image_kernel_col_major_zero_matrix() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_column(DynamicVector::from(vec![0.0, 0.0]));
    m.append_column(DynamicVector::from(vec![0.0, 0.0])); // 2x2 zero matrix

    let image = m.image();
    assert_eq!(image.len(), 0, "Image of zero matrix should be trivial");

    let kernel = m.kernel();
    let expected_kernel_basis =
      [DynamicVector::from(vec![1.0, 0.0]), DynamicVector::from(vec![0.0, 1.0])];
    assert_eq!(kernel.len(), 2);
    assert!(contains_vector(&kernel, &expected_kernel_basis[0]));
    assert!(contains_vector(&kernel, &expected_kernel_basis[1]));
  }

  #[test]
  fn test_empty_matrix_0x0_row_major() {
    let m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    assert_eq!(m.num_rows(), 0);
    assert_eq!(m.num_cols(), 0);
    assert_eq!(m.image().len(), 0);
    assert_eq!(m.kernel().len(), 0);
  }

  #[test]
  fn test_matrix_3x0_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::from(vec![]));
    m.append_row(DynamicVector::from(vec![]));
    m.append_row(DynamicVector::from(vec![]));
    assert_eq!(m.num_rows(), 3);
    assert_eq!(m.num_cols(), 0);
    assert_eq!(m.image().len(), 0);
    assert_eq!(m.kernel().len(), 0);
  }

  #[test]
  fn test_empty_matrix_0x0_col_major() {
    let m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    assert_eq!(m.num_rows(), 0);
    assert_eq!(m.num_cols(), 0);
    assert_eq!(m.image().len(), 0);
    assert_eq!(m.kernel().len(), 0);
  }

  #[test]
  fn test_matrix_0x3_col_major() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_column(DynamicVector::from(vec![]));
    m.append_column(DynamicVector::from(vec![]));
    m.append_column(DynamicVector::from(vec![]));
    assert_eq!(m.num_rows(), 0);
    assert_eq!(m.num_cols(), 3);
    assert_eq!(m.image().len(), 0);
    // Kernel for 0xN matrix (A x = 0 always true) is R^N
    let kernel = m.kernel();
    assert_eq!(kernel.len(), 3);
    assert!(contains_vector(&kernel, &DynamicVector::from(vec![1.0, 0.0, 0.0])));
    assert!(contains_vector(&kernel, &DynamicVector::from(vec![0.0, 1.0, 0.0])));
    assert!(contains_vector(&kernel, &DynamicVector::from(vec![0.0, 0.0, 1.0])));
  }

  #[test]
  fn test_matrix_vector_mul_row_major() {
    let mut m: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    m.append_row(DynamicVector::from(vec![1.0, 2.0, 3.0]));
    m.append_row(DynamicVector::from(vec![4.0, 5.0, 6.0]));
    m.append_row(DynamicVector::from(vec![7.0, 8.0, 9.0]));
    let v = DynamicVector::from(vec![1.0, 2.0, 3.0]);
    let result = m * v;
    assert_eq!(result, DynamicVector::from(vec![14.0, 32.0, 50.0]));
  }

  #[test]
  fn test_matrix_vector_mul_col_major() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    m.append_row(&DynamicVector::from(vec![1.0, 2.0, 3.0]));
    m.append_row(&DynamicVector::from(vec![4.0, 5.0, 6.0]));
    m.append_row(&DynamicVector::from(vec![7.0, 8.0, 9.0]));
    let v = DynamicVector::from(vec![1.0, 2.0, 3.0]);
    let result = m * v;
    assert_eq!(result, DynamicVector::from(vec![14.0, 32.0, 50.0]));
  }

  #[test]
  fn test_matrix_zeros() {
    let m = DynamicDenseMatrix::<f64, RowMajor>::zeros(2, 3);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 3);
    assert_eq!(m.image().len(), 0);
    assert_eq!(m.kernel().len(), 3);

    let m = DynamicDenseMatrix::<f64, ColumnMajor>::zeros(2, 3);
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 3);
    assert_eq!(m.image().len(), 0);
    assert_eq!(m.kernel().len(), 3);
  }

  #[test]
  fn test_matrix_matmul() {
    let mut m: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    // m (CM, 2x3)
    // 1  2  3
    // 4  5  6
    m.append_column(DynamicVector::from(vec![1.0, 4.0]));
    m.append_column(DynamicVector::from(vec![2.0, 5.0]));
    m.append_column(DynamicVector::from(vec![3.0, 6.0]));

    let mut n: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    // n (RM, 3x2)
    // 9  10
    // 11 12
    // 13 14
    n.append_row(DynamicVector::from(vec![9.0, 10.0]));
    n.append_row(DynamicVector::from(vec![11.0, 12.0]));
    n.append_row(DynamicVector::from(vec![13.0, 14.0]));

    // m (CM 2x3) * n (RM 3x2) = result (RM 2x2)
    let result = m * n;
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_cols(), 2);
    // Expected:
    // (1*9 + 2*11 + 3*13) (1*10 + 2*12 + 3*14) = (9+22+39) (10+24+42) = (70) (76)
    // (4*9 + 5*11 + 6*13) (4*10 + 5*12 + 6*14) = (36+55+78) (40+60+84) = (169) (184)
    assert_eq!(*result.get_component(0, 0), 1.0 * 9.0 + 2.0 * 11.0 + 3.0 * 13.0);
    assert_eq!(*result.get_component(0, 1), 1.0 * 10.0 + 2.0 * 12.0 + 3.0 * 14.0);
    assert_eq!(*result.get_component(1, 0), 4.0 * 9.0 + 5.0 * 11.0 + 6.0 * 13.0);
    assert_eq!(*result.get_component(1, 1), 4.0 * 10.0 + 5.0 * 12.0 + 6.0 * 14.0);
  }

  #[test]
  fn test_matrix_matmul_rm_rm() {
    // A (RM 2x2)
    // 1 2
    // 3 4
    let mut a_rm: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    a_rm.append_row(DynamicVector::from(vec![1.0, 2.0]));
    a_rm.append_row(DynamicVector::from(vec![3.0, 4.0]));

    // B (RM 2x2)
    // 5 6
    // 7 8
    let mut b_rm: DynamicDenseMatrix<f64, RowMajor> = DynamicDenseMatrix::new();
    b_rm.append_row(DynamicVector::from(vec![5.0, 6.0]));
    b_rm.append_row(DynamicVector::from(vec![7.0, 8.0]));

    // Expected A * B (RM 2x2)
    // 19 22
    // 43 50
    let result = a_rm * b_rm;
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_cols(), 2);
    assert_eq!(*result.get_component(0, 0), 1.0 * 5.0 + 2.0 * 7.0);
    assert_eq!(*result.get_component(0, 1), 1.0 * 6.0 + 2.0 * 8.0);
    assert_eq!(*result.get_component(1, 0), 3.0 * 5.0 + 4.0 * 7.0);
    assert_eq!(*result.get_component(1, 1), 3.0 * 6.0 + 4.0 * 8.0);
  }

  #[test]
  fn test_matrix_matmul_cm_cm() {
    // A (CM 2x2)
    // 1 2
    // 3 4
    let mut a_cm: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    a_cm.append_column(DynamicVector::from(vec![1.0, 3.0]));
    a_cm.append_column(DynamicVector::from(vec![2.0, 4.0]));

    // B (CM 2x2)
    // 5 6
    // 7 8
    let mut b_cm: DynamicDenseMatrix<f64, ColumnMajor> = DynamicDenseMatrix::new();
    b_cm.append_column(DynamicVector::from(vec![5.0, 7.0]));
    b_cm.append_column(DynamicVector::from(vec![6.0, 8.0]));

    // A (CM 2x2) * B (CM 2x2) = result (CM 2x2)
    // If CM*CM impl is A*B:
    // Expected A * B (CM 2x2)
    // 19 22
    // 43 50
    // If CM*CM impl is B*A^T (as suspected from code reading):
    // B (CM 2x2) * A^T (RM 2x2)
    // A^T (RM):
    // 1 3
    // 2 4
    // B * A^T (CM * RM -> RM result, but CM*CM -> CM result. The code seems to produce (B*A^T)
    // stored as CM) (5*1 + 6*2) (5*3 + 6*4) = (5+12) (15+24) = 17 39
    // (7*1 + 8*2) (7*3 + 8*4) = (7+16) (21+32) = 23 53
    // Expected if B*A^T, stored as CM:
    // 17 23
    // 39 53

    let result = a_cm * b_cm; // Output is ColumnMajor
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_cols(), 2);

    // Assuming standard A*B for now. If this fails, the impl is non-standard.
    assert_eq!(*result.get_component(0, 0), 1.0 * 5.0 + 2.0 * 7.0); // row 0, col 0
    assert_eq!(*result.get_component(0, 1), 1.0 * 6.0 + 2.0 * 8.0); // row 0, col 1
    assert_eq!(*result.get_component(1, 0), 3.0 * 5.0 + 4.0 * 7.0); // row 1, col 0
    assert_eq!(*result.get_component(1, 1), 3.0 * 6.0 + 4.0 * 8.0); // row 1, col 1
  }
}
