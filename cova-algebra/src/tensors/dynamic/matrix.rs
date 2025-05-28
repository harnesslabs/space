//! # Dynamic Matrix Module
//!
//! This module provides a clean, ergonomic implementation of matrices with dynamically determined
//! dimensions. The design prioritizes simplicity and ease of use while maintaining mathematical
//! correctness and performance.
//!
//! ## Design Philosophy
//!
//! This implementation favors **simplicity over micro-optimizations**. Rather than exposing
//! storage orientation complexity to users, we provide a single, well-optimized matrix type
//! with intuitive builder patterns and method chaining.
//!
//! ## Mathematical Background
//!
//! A matrix is a rectangular array of elements arranged in rows and columns. For a matrix $A$
//! with $m$ rows and $n$ columns, we write $A \in F^{m \times n}$ where $F$ is the field of
//! the matrix elements.
//!
//! ## Examples
//!
//! ```
//! use cova_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{matrix::Matrix, vector::Vector},
//! };
//!
//! // Create matrices using builder pattern
//! let matrix = Matrix::builder().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).build();
//!
//! // Or from vectors
//! let matrix = Matrix::from_rows([Vector::from([1.0, 2.0, 3.0]), Vector::from([4.0, 5.0, 6.0])]);
//!
//! // Common constructors
//! let zeros = Matrix::<f64>::zeros(3, 3);
//! let identity = Matrix::<f64>::identity(3);
//!
//! // Access elements naturally
//! let element = matrix[(0, 1)]; // Gets element at row 0, column 1
//!
//! // Linear algebra operations
//! let rref = matrix.clone().into_row_echelon_form();
//! let kernel = matrix.kernel();
//! let image = matrix.image();
//!
//! // Block matrix construction
//! let block_matrix = Matrix::from_blocks(vec![
//!   vec![Matrix::<f64>::identity(2), Matrix::<f64>::zeros(2, 3)],
//!   vec![Matrix::<f64>::zeros(3, 2), Matrix::<f64>::identity(3)],
//! ]);
//!
//! // Or using block builder
//! let block_matrix = Matrix::<f64>::block_builder()
//!   .block(0, 0, Matrix::<f64>::identity(2))
//!   .block(1, 1, Matrix::<f64>::identity(3))
//!   .build(vec![2, 3], vec![2, 3]);
//!
//! // For more complex block structures:
//! let blocks = vec![vec![Matrix::<f64>::identity(2), Matrix::<f64>::zeros(2, 3)], vec![
//!   Matrix::<f64>::zeros(3, 2),
//!   Matrix::<f64>::identity(3),
//! ]];
//! let complex_block_matrix = Matrix::from_blocks(blocks);
//!
//! // Display the matrix
//! println!("{}", block_matrix);
//!
//! // Create a 3x3 block matrix
//! let a = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();
//! let b = Matrix::builder().row([5.0]).row([6.0]).build();
//! let c = Matrix::builder().row([7.0, 8.0]).build();
//! let d = Matrix::builder().row([9.0]).build();
//! let e = Matrix::<f64>::identity(2);
//! let f = Matrix::builder().row([10.0, 11.0]).build();
//!
//! let blocks = vec![vec![a, b, Matrix::<f64>::zeros(2, 2)], vec![c, d, f], vec![
//!   Matrix::<f64>::zeros(2, 2),
//!   Matrix::<f64>::zeros(2, 1),
//!   e,
//! ]];
//! let complex_block_matrix = Matrix::from_blocks(blocks);
//!
//! println!("{}", block_matrix);
//! ```

use std::{fmt, ops::Index};

use super::{vector::Vector, *};

/// Information about a pivot (non-zero entry) in a matrix.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PivotInfo {
  /// The row index of the pivot
  pub row: usize,
  /// The column index of the pivot
  pub col: usize,
}

/// Result of transforming a matrix to row echelon form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEchelonOutput {
  /// The rank of the matrix (number of linearly independent rows/columns)
  pub rank:   usize,
  /// Positions of pivot elements in the row echelon form
  pub pivots: Vec<PivotInfo>,
}

/// A dynamically-sized matrix with elements from a field `F`.
///
/// This implementation uses row-major storage internally but provides an intuitive,
/// orientation-agnostic API. The focus is on ergonomics and correctness rather than
/// micro-optimizations for specific access patterns.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Matrix<F> {
  /// Internal storage as a vector of row vectors
  rows: Vec<Vector<F>>,
}

impl<F> Matrix<F> {
  /// Creates a new empty matrix.
  pub const fn new() -> Self { Self { rows: Vec::new() } }

  /// Returns the number of rows in the matrix.
  pub fn num_rows(&self) -> usize { self.rows.len() }

  /// Returns the number of columns in the matrix.
  pub fn num_cols(&self) -> usize { self.rows.first().map_or(0, |row| row.dimension()) }

  /// Returns the dimensions as (rows, cols).
  pub fn dimensions(&self) -> (usize, usize) { (self.num_rows(), self.num_cols()) }

  /// Checks if the matrix is empty (no rows or no columns).
  pub fn is_empty(&self) -> bool { self.num_rows() == 0 || self.num_cols() == 0 }

  /// Gets a reference to the component at the given row and column.
  pub fn get(&self, row: usize, col: usize) -> Option<&F> {
    self.rows.get(row)?.get_component(col).into()
  }

  /// Gets a reference to the row at the given index.
  pub fn row(&self, index: usize) -> Option<&Vector<F>> { self.rows.get(index) }

  /// Returns an iterator over the rows.
  pub fn rows(&self) -> impl Iterator<Item = &Vector<F>> { self.rows.iter() }
}

impl<F: Field + Copy> Matrix<F> {
  /// Creates a matrix filled with zeros.
  pub fn zeros(rows: usize, cols: usize) -> Self {
    Self { rows: (0..rows).map(|_| Vector::zeros(cols)).collect() }
  }

  /// Creates an identity matrix of the given size.
  pub fn identity(size: usize) -> Self {
    let mut matrix = Self::zeros(size, size);
    for i in 0..size {
      matrix.set(i, i, F::one());
    }
    matrix
  }

  /// Creates a matrix from an iterator of rows.
  pub fn from_rows<I>(rows: I) -> Self
  where I: IntoIterator<Item = Vector<F>> {
    let rows: Vec<_> = rows.into_iter().collect();

    // Validate that all rows have the same length
    if let Some(expected_cols) = rows.first().map(|r| r.dimension()) {
      for (i, row) in rows.iter().enumerate() {
        assert_eq!(
          row.dimension(),
          expected_cols,
          "Row {} has {} columns, expected {}",
          i,
          row.dimension(),
          expected_cols
        );
      }
    }

    Self { rows }
  }

  /// Creates a matrix from an iterator of columns.
  pub fn from_cols<I>(cols: I) -> Self
  where I: IntoIterator<Item = Vector<F>> {
    let cols: Vec<_> = cols.into_iter().collect();

    if cols.is_empty() {
      return Self::new();
    }

    let num_rows = cols[0].dimension();
    let num_cols = cols.len();

    // Validate all columns have same length
    for (i, col) in cols.iter().enumerate() {
      assert_eq!(
        col.dimension(),
        num_rows,
        "Column {} has {} rows, expected {}",
        i,
        col.dimension(),
        num_rows
      );
    }

    let mut matrix = Self::zeros(num_rows, num_cols);
    for (col_idx, col) in cols.iter().enumerate() {
      for (row_idx, &value) in col.components().iter().enumerate() {
        matrix.set(row_idx, col_idx, value);
      }
    }

    matrix
  }

  /// Creates a matrix from a 2D vector of blocks.
  ///
  /// All blocks in the same block-row must have the same height, and all blocks in the
  /// same block-column must have the same width. Use `Matrix::zeros(rows, cols)` for
  /// zero blocks.
  ///
  /// # Arguments
  ///
  /// * `blocks`: 2D vector where `blocks[i][j]` is the block at block-row i, block-column j
  ///
  /// # Examples
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// // Create a 2x2 block matrix with identity blocks on the diagonal
  /// let blocks = vec![vec![Matrix::<f64>::identity(2), Matrix::<f64>::zeros(2, 3)], vec![
  ///   Matrix::<f64>::zeros(3, 2),
  ///   Matrix::<f64>::identity(3),
  /// ]];
  /// let block_matrix = Matrix::from_blocks(blocks);
  ///
  /// println!("{}", block_matrix);
  /// ```
  ///
  /// For more complex block structures:
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// // Create a 3x3 block matrix
  /// let a = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();
  /// let b = Matrix::builder().row([5.0]).row([6.0]).build();
  /// let c = Matrix::builder().row([7.0, 8.0]).build();
  /// let d = Matrix::builder().row([9.0]).build();
  /// let e = Matrix::<f64>::identity(2);
  /// let f = Matrix::builder().row([10.0, 11.0]).build();
  ///
  /// let blocks = vec![vec![a, b, Matrix::<f64>::zeros(2, 2)], vec![c, d, f], vec![
  ///   Matrix::<f64>::zeros(2, 2),
  ///   Matrix::<f64>::zeros(2, 1),
  ///   e,
  /// ]];
  /// let complex_block_matrix = Matrix::from_blocks(blocks);
  /// ```
  pub fn from_blocks(blocks: Vec<Vec<Matrix<F>>>) -> Self {
    if blocks.is_empty() || blocks[0].is_empty() {
      return Self::new();
    }

    let block_rows = blocks.len();
    let block_cols = blocks[0].len();

    // Validate that all rows have the same number of columns
    for (i, row) in blocks.iter().enumerate() {
      assert_eq!(
        row.len(),
        block_cols,
        "Block row {} has {} columns, expected {}",
        i,
        row.len(),
        block_cols
      );
    }

    // Determine block sizes by examining blocks in each row/column
    let mut row_block_sizes = vec![0; block_rows];
    let mut col_block_sizes = vec![0; block_cols];

    // Find row heights from the first column
    for (block_row, row) in blocks.iter().enumerate() {
      row_block_sizes[block_row] = row[0].num_rows();
    }

    // Find column widths from the first row
    for (block_col, block) in blocks[0].iter().enumerate() {
      col_block_sizes[block_col] = block.num_cols();
    }

    // Validate that all blocks have consistent dimensions
    for (block_row, row) in blocks.iter().enumerate() {
      for (block_col, block) in row.iter().enumerate() {
        assert_eq!(
          block.num_rows(),
          row_block_sizes[block_row],
          "Block at ({}, {}) has {} rows, expected {}",
          block_row,
          block_col,
          block.num_rows(),
          row_block_sizes[block_row]
        );
        assert_eq!(
          block.num_cols(),
          col_block_sizes[block_col],
          "Block at ({}, {}) has {} columns, expected {}",
          block_row,
          block_col,
          block.num_cols(),
          col_block_sizes[block_col]
        );
      }
    }

    // Calculate total dimensions
    let total_rows: usize = row_block_sizes.iter().sum();
    let total_cols: usize = col_block_sizes.iter().sum();

    let mut result = Self::zeros(total_rows, total_cols);

    // Compute offsets for efficient placement
    let mut row_offsets = vec![0; block_rows + 1];
    for i in 0..block_rows {
      row_offsets[i + 1] = row_offsets[i] + row_block_sizes[i];
    }

    let mut col_offsets = vec![0; block_cols + 1];
    for i in 0..block_cols {
      col_offsets[i + 1] = col_offsets[i] + col_block_sizes[i];
    }

    // Place blocks
    for (block_row, row) in blocks.iter().enumerate() {
      for (block_col, block) in row.iter().enumerate() {
        let row_start = row_offsets[block_row];
        let col_start = col_offsets[block_col];

        for i in 0..block.num_rows() {
          for j in 0..block.num_cols() {
            result.set(row_start + i, col_start + j, *block.get(i, j).unwrap());
          }
        }
      }
    }

    result
  }

  /// Sets the component at the given row and column.
  pub fn set(&mut self, row: usize, col: usize, value: F) {
    assert!(row < self.num_rows(), "Row index {row} out of bounds");
    assert!(col < self.num_cols(), "Column index {col} out of bounds");
    self.rows[row].set_component(col, value);
  }

  /// Appends a row to the matrix.
  pub fn push_row(&mut self, row: Vector<F>) {
    if !self.rows.is_empty() {
      assert_eq!(
        row.dimension(),
        self.num_cols(),
        "Row has {} columns, expected {}",
        row.dimension(),
        self.num_cols()
      );
    }
    self.rows.push(row);
  }

  /// Appends a column to the matrix.
  pub fn push_column(&mut self, col: &Vector<F>) {
    if self.is_empty() {
      // If matrix is empty, create rows from the column
      for &component in col.components() {
        self.rows.push(Vector::from([component]));
      }
    } else {
      assert_eq!(
        col.dimension(),
        self.num_rows(),
        "Column has {} rows, expected {}",
        col.dimension(),
        self.num_rows()
      );

      for (row_idx, &component) in col.components().iter().enumerate() {
        self.rows[row_idx].append(component);
      }
    }
  }

  /// Gets the column at the given index as a new vector.
  pub fn column(&self, index: usize) -> Vector<F> {
    assert!(index < self.num_cols(), "Column index {index} out of bounds");

    let components: Vec<F> = self.rows.iter().map(|row| *row.get_component(index)).collect();

    Vector::new(components)
  }

  /// Sets the column at the given index.
  pub fn set_column(&mut self, index: usize, column: &Vector<F>) {
    assert!(index < self.num_cols(), "Column index {index} out of bounds");
    assert_eq!(
      column.dimension(),
      self.num_rows(),
      "Column has {} rows, expected {}",
      column.dimension(),
      self.num_rows()
    );

    for (row_idx, &value) in column.components().iter().enumerate() {
      self.rows[row_idx].set_component(index, value);
    }
  }

  /// Transposes the matrix, returning a new matrix.
  pub fn transpose(&self) -> Self {
    if self.is_empty() {
      return Self::new();
    }

    let cols: Vec<_> = (0..self.num_cols()).map(|i| self.column(i)).collect();

    Self::from_rows(cols)
  }

  /// Transforms this matrix into row echelon form in-place.
  pub fn into_row_echelon_form(mut self) -> (Self, RowEchelonOutput) {
    let output = self.row_echelon_form_inplace();
    (self, output)
  }

  /// Transforms this matrix into row echelon form in-place, returning pivot information.
  fn row_echelon_form_inplace(&mut self) -> RowEchelonOutput {
    if self.is_empty() {
      return RowEchelonOutput { rank: 0, pivots: Vec::new() };
    }

    let rows = self.num_rows();
    let cols = self.num_cols();
    let mut lead = 0;
    let mut rank = 0;
    let mut pivots = Vec::new();

    for r in 0..rows {
      if lead >= cols {
        break;
      }

      let mut i = r;
      while self.rows[i].get_component(lead).is_zero() {
        i += 1;
        if i == rows {
          i = r;
          lead += 1;
          if lead == cols {
            return RowEchelonOutput { rank, pivots };
          }
        }
      }

      // Swap rows
      self.rows.swap(i, r);

      pivots.push(PivotInfo { row: r, col: lead });

      let pivot_val = *self.rows[r].get_component(lead);
      let inv_pivot = pivot_val.multiplicative_inverse();

      // Scale pivot row
      for j in lead..cols {
        let val = *self.rows[r].get_component(j);
        self.rows[r].set_component(j, val * inv_pivot);
      }

      // Eliminate column
      for i_row in 0..rows {
        if i_row != r {
          let factor = *self.rows[i_row].get_component(lead);
          if !factor.is_zero() {
            for j_col in lead..cols {
              let val_r = *self.rows[r].get_component(j_col);
              let val_i = *self.rows[i_row].get_component(j_col);
              self.rows[i_row].set_component(j_col, val_i - factor * val_r);
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
  pub fn image(&self) -> Vec<Vector<F>> {
    if self.is_empty() {
      return Vec::new();
    }

    let (_, echelon_output) = self.clone().into_row_echelon_form();
    let pivot_cols: Vec<usize> = echelon_output.pivots.iter().map(|p| p.col).collect();

    pivot_cols.into_iter().map(|col_idx| self.column(col_idx)).collect()
  }

  /// Computes a basis for the kernel (null space) of the matrix.
  pub fn kernel(&self) -> Vec<Vector<F>> {
    if self.num_cols() == 0 {
      return Vec::new();
    }

    if self.num_rows() == 0 {
      // All vectors are in the kernel
      return (0..self.num_cols())
        .map(|i| {
          let mut components = vec![F::zero(); self.num_cols()];
          components[i] = F::one();
          Vector::new(components)
        })
        .collect();
    }

    let (rref_matrix, echelon_output) = self.clone().into_row_echelon_form();
    let num_cols = self.num_cols();

    let mut is_pivot_col = vec![false; num_cols];
    for pivot in &echelon_output.pivots {
      if pivot.col < num_cols {
        is_pivot_col[pivot.col] = true;
      }
    }

    let free_cols: Vec<usize> = (0..num_cols).filter(|&j| !is_pivot_col[j]).collect();

    let mut kernel_basis = Vec::new();

    for &free_idx in &free_cols {
      let mut basis_vector = vec![F::zero(); num_cols];
      basis_vector[free_idx] = F::one();

      for pivot in &echelon_output.pivots {
        let pivot_col = pivot.col;
        let pivot_row = pivot.row;

        if pivot_col < num_cols && pivot_row < rref_matrix.num_rows() {
          let coefficient = *rref_matrix.rows[pivot_row].get_component(free_idx);
          basis_vector[pivot_col] = -coefficient;
        }
      }

      kernel_basis.push(Vector::new(basis_vector));
    }

    kernel_basis
  }

  /// Creates a matrix builder for fluent construction.
  pub fn builder() -> MatrixBuilder<F> { MatrixBuilder::new() }

  /// Creates a block matrix builder for fluent block construction.
  pub fn block_builder() -> BlockMatrixBuilder<F> { BlockMatrixBuilder::new() }

  /// Extracts a block from the matrix at the specified position.
  ///
  /// # Arguments
  ///
  /// * `row_start`: Starting row index (inclusive)
  /// * `row_end`: Ending row index (exclusive)
  /// * `col_start`: Starting column index (inclusive)
  /// * `col_end`: Ending column index (exclusive)
  ///
  /// # Examples
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// let matrix = Matrix::builder()
  ///   .row([1.0, 2.0, 3.0, 4.0])
  ///   .row([5.0, 6.0, 7.0, 8.0])
  ///   .row([9.0, 10.0, 11.0, 12.0])
  ///   .build();
  ///
  /// let block = matrix.extract_block(0, 2, 1, 3);
  /// // block is now [[2.0, 3.0], [6.0, 7.0]]
  /// ```
  pub fn extract_block(
    &self,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
  ) -> Self {
    assert!(row_start <= row_end, "row_start must be <= row_end");
    assert!(col_start <= col_end, "col_start must be <= col_end");
    assert!(row_end <= self.num_rows(), "row_end out of bounds");
    assert!(col_end <= self.num_cols(), "col_end out of bounds");

    let block_rows = row_end - row_start;
    let block_cols = col_end - col_start;

    if block_rows == 0 || block_cols == 0 {
      return Self::new();
    }

    let mut result = Self::zeros(block_rows, block_cols);

    for i in 0..block_rows {
      for j in 0..block_cols {
        let value = *self.get(row_start + i, col_start + j).unwrap();
        result.set(i, j, value);
      }
    }

    result
  }

  /// Sets a block in the matrix at the specified position.
  ///
  /// # Arguments
  ///
  /// * `row_start`: Starting row index where the block should be placed
  /// * `col_start`: Starting column index where the block should be placed
  /// * `block`: The matrix block to insert
  ///
  /// # Panics
  ///
  /// Panics if the block would extend beyond the matrix boundaries.
  pub fn set_block(&mut self, row_start: usize, col_start: usize, block: &Matrix<F>) {
    assert!(
      row_start + block.num_rows() <= self.num_rows(),
      "Block would extend beyond matrix row boundary"
    );
    assert!(
      col_start + block.num_cols() <= self.num_cols(),
      "Block would extend beyond matrix column boundary"
    );

    for i in 0..block.num_rows() {
      for j in 0..block.num_cols() {
        let value = *block.get(i, j).unwrap();
        self.set(row_start + i, col_start + j, value);
      }
    }
  }

  /// Creates a block diagonal matrix from a list of matrices.
  ///
  /// # Examples
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// let blocks = vec![Matrix::<f64>::identity(2), Matrix::<f64>::identity(3)];
  ///
  /// let block_diag = Matrix::block_diagonal(blocks);
  /// // Creates a 5x5 matrix with 2x2 and 3x3 identity blocks on the diagonal
  /// ```
  pub fn block_diagonal<I>(blocks: I) -> Self
  where I: IntoIterator<Item = Matrix<F>> {
    let blocks: Vec<_> = blocks.into_iter().collect();

    if blocks.is_empty() {
      return Self::new();
    }

    let total_rows: usize = blocks.iter().map(|b| b.num_rows()).sum();
    let total_cols: usize = blocks.iter().map(|b| b.num_cols()).sum();

    let mut result = Self::zeros(total_rows, total_cols);

    let mut row_offset = 0;
    let mut col_offset = 0;

    for block in &blocks {
      result.set_block(row_offset, col_offset, block);
      row_offset += block.num_rows();
      col_offset += block.num_cols();
    }

    result
  }

  /// Creates a matrix by horizontally concatenating (stacking side by side) matrices.
  ///
  /// All matrices must have the same number of rows.
  pub fn hstack<I>(matrices: I) -> Self
  where I: IntoIterator<Item = Matrix<F>> {
    let matrices: Vec<_> = matrices.into_iter().collect();

    if matrices.is_empty() {
      return Self::new();
    }

    let num_rows = matrices[0].num_rows();
    for (i, matrix) in matrices.iter().enumerate() {
      assert_eq!(
        matrix.num_rows(),
        num_rows,
        "Matrix {} has {} rows, expected {}",
        i,
        matrix.num_rows(),
        num_rows
      );
    }

    let total_cols: usize = matrices.iter().map(|m| m.num_cols()).sum();
    let mut result = Self::zeros(num_rows, total_cols);

    let mut col_offset = 0;
    for matrix in &matrices {
      result.set_block(0, col_offset, matrix);
      col_offset += matrix.num_cols();
    }

    result
  }

  /// Creates a matrix by vertically concatenating (stacking on top of each other) matrices.
  ///
  /// All matrices must have the same number of columns.
  pub fn vstack<I>(matrices: I) -> Self
  where I: IntoIterator<Item = Matrix<F>> {
    let matrices: Vec<_> = matrices.into_iter().collect();

    if matrices.is_empty() {
      return Self::new();
    }

    let num_cols = matrices[0].num_cols();
    for (i, matrix) in matrices.iter().enumerate() {
      assert_eq!(
        matrix.num_cols(),
        num_cols,
        "Matrix {} has {} columns, expected {}",
        i,
        matrix.num_cols(),
        num_cols
      );
    }

    let total_rows: usize = matrices.iter().map(|m| m.num_rows()).sum();
    let mut result = Self::zeros(total_rows, num_cols);

    let mut row_offset = 0;
    for matrix in &matrices {
      result.set_block(row_offset, 0, matrix);
      row_offset += matrix.num_rows();
    }

    result
  }
}

// Index trait for convenient element access
impl<F> Index<(usize, usize)> for Matrix<F> {
  type Output = F;

  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    self.get(row, col).expect("Index out of bounds")
  }
}

/// Builder for constructing matrices with a fluent API.
pub struct MatrixBuilder<F> {
  rows:       Vec<Vector<F>>,
  cols:       Vec<Vector<F>>,
  build_mode: BuildMode,
}

#[derive(Debug, Clone, PartialEq)]
enum BuildMode {
  Rows,
  Columns,
}

impl<F: Field + Copy> MatrixBuilder<F> {
  /// Creates a new matrix builder.
  pub fn new() -> Self { Self { rows: Vec::new(), cols: Vec::new(), build_mode: BuildMode::Rows } }

  /// Adds a row from an array.
  pub fn row<const N: usize>(mut self, row: [F; N]) -> Self {
    assert_eq!(
      self.build_mode,
      BuildMode::Rows,
      "Cannot mix row and column operations in MatrixBuilder"
    );
    self.rows.push(Vector::from(row));
    self
  }

  /// Adds a row from a vector.
  pub fn row_vec(mut self, row: Vector<F>) -> Self {
    assert_eq!(
      self.build_mode,
      BuildMode::Rows,
      "Cannot mix row and column operations in MatrixBuilder"
    );
    self.rows.push(row);
    self
  }

  /// Adds a row from an iterator.
  pub fn row_iter<I>(mut self, row: I) -> Self
  where I: IntoIterator<Item = F> {
    assert_eq!(
      self.build_mode,
      BuildMode::Rows,
      "Cannot mix row and column operations in MatrixBuilder"
    );
    self.rows.push(Vector::from(row.into_iter().collect::<Vec<_>>()));
    self
  }

  /// Adds a column from an array.
  pub fn column<const N: usize>(mut self, col: [F; N]) -> Self {
    if self.build_mode == BuildMode::Rows && !self.rows.is_empty() {
      panic!("Cannot mix row and column operations in MatrixBuilder");
    }
    self.build_mode = BuildMode::Columns;
    self.cols.push(Vector::from(col));
    self
  }

  /// Adds a column from a vector.
  pub fn column_vec(mut self, col: Vector<F>) -> Self {
    if self.build_mode == BuildMode::Rows && !self.rows.is_empty() {
      panic!("Cannot mix row and column operations in MatrixBuilder");
    }
    self.build_mode = BuildMode::Columns;
    self.cols.push(col);
    self
  }

  /// Adds a column from an iterator.
  pub fn column_iter<I>(mut self, col: I) -> Self
  where I: IntoIterator<Item = F> {
    if self.build_mode == BuildMode::Rows && !self.rows.is_empty() {
      panic!("Cannot mix row and column operations in MatrixBuilder");
    }
    self.build_mode = BuildMode::Columns;
    self.cols.push(Vector::from(col.into_iter().collect::<Vec<_>>()));
    self
  }

  /// Builds the matrix.
  pub fn build(self) -> Matrix<F> {
    match self.build_mode {
      BuildMode::Rows => Matrix::from_rows(self.rows),
      BuildMode::Columns => Matrix::from_cols(self.cols),
    }
  }
}

impl<F: Field + Copy> Default for MatrixBuilder<F> {
  fn default() -> Self { Self::new() }
}

/// Builder for constructing block matrices with a fluent API.
///
/// This builder allows you to specify blocks at specific positions and then
/// build the final matrix with the specified block structure. The builder
/// implements `Display` to show the current construction state, making it
/// easy to debug block matrix construction.
///
/// # Examples
///
/// ```
/// use cova_algebra::tensors::dynamic::matrix::Matrix;
///
/// let builder = Matrix::<f64>::block_builder().block(0, 0, Matrix::<f64>::identity(2)).block(
///   1,
///   1,
///   Matrix::<f64>::identity(3),
/// );
///
/// // Debug the construction state
/// println!("Builder state: {}", builder);
///
/// // Build the final matrix
/// let block_matrix = builder.build(vec![2, 3], vec![2, 3]);
/// ```
pub struct BlockMatrixBuilder<F> {
  blocks: std::collections::HashMap<(usize, usize), Matrix<F>>,
}

impl<F: Field + Copy> BlockMatrixBuilder<F> {
  /// Creates a new block matrix builder.
  pub fn new() -> Self { Self { blocks: std::collections::HashMap::new() } }

  /// Adds a block at the specified position.
  pub fn block(mut self, block_row: usize, block_col: usize, matrix: Matrix<F>) -> Self {
    self.blocks.insert((block_row, block_col), matrix);
    self
  }

  /// Builds the block matrix with the specified block structure.
  ///
  /// # Arguments
  ///
  /// * `row_block_sizes`: Vector specifying the height of each block row
  /// * `col_block_sizes`: Vector specifying the width of each block column
  ///
  /// # Panics
  ///
  /// Panics if any block has dimensions that don't match the specified structure.
  ///
  /// # Examples
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// let builder = Matrix::<f64>::block_builder().block(0, 0, Matrix::<f64>::identity(2)).block(
  ///   1,
  ///   1,
  ///   Matrix::<f64>::identity(3),
  /// );
  ///
  /// // Debug the construction state
  /// println!("Builder state: {}", builder);
  ///
  /// // Build the final matrix
  /// let block_matrix = builder.build(vec![2, 3], vec![2, 3]);
  /// ```
  pub fn build(self, row_block_sizes: Vec<usize>, col_block_sizes: Vec<usize>) -> Matrix<F> {
    let block_rows = row_block_sizes.len();
    let block_cols = col_block_sizes.len();

    // Validate block dimensions
    for ((block_row, block_col), matrix) in &self.blocks {
      assert!(
        *block_row < block_rows,
        "Block row {} out of bounds (max: {})",
        block_row,
        block_rows - 1
      );
      assert!(
        *block_col < block_cols,
        "Block column {} out of bounds (max: {})",
        block_col,
        block_cols - 1
      );

      let expected_rows = row_block_sizes[*block_row];
      let expected_cols = col_block_sizes[*block_col];

      assert_eq!(
        matrix.num_rows(),
        expected_rows,
        "Block at ({}, {}) has {} rows, expected {}",
        block_row,
        block_col,
        matrix.num_rows(),
        expected_rows
      );
      assert_eq!(
        matrix.num_cols(),
        expected_cols,
        "Block at ({}, {}) has {} columns, expected {}",
        block_row,
        block_col,
        matrix.num_cols(),
        expected_cols
      );
    }

    // Calculate total dimensions
    let total_rows: usize = row_block_sizes.iter().sum();
    let total_cols: usize = col_block_sizes.iter().sum();

    let mut result = Matrix::zeros(total_rows, total_cols);

    // Compute offsets
    let mut row_offsets = vec![0; block_rows + 1];
    for i in 0..block_rows {
      row_offsets[i + 1] = row_offsets[i] + row_block_sizes[i];
    }

    let mut col_offsets = vec![0; block_cols + 1];
    for i in 0..block_cols {
      col_offsets[i + 1] = col_offsets[i] + col_block_sizes[i];
    }

    // Place blocks
    for ((block_row, block_col), matrix) in &self.blocks {
      let row_start = row_offsets[*block_row];
      let col_start = col_offsets[*block_col];

      for i in 0..matrix.num_rows() {
        for j in 0..matrix.num_cols() {
          result.set(row_start + i, col_start + j, *matrix.get(i, j).unwrap());
        }
      }
    }

    result
  }
}

impl<F: Field + Copy> Default for BlockMatrixBuilder<F> {
  fn default() -> Self { Self::new() }
}

impl<F: Field + Copy + fmt::Display> fmt::Display for BlockMatrixBuilder<F> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.blocks.is_empty() {
      return writeln!(f, "BlockMatrixBuilder: (empty)");
    }

    // Determine the grid dimensions by finding max block coordinates
    let max_block_row = self.blocks.keys().map(|(r, _)| *r).max().unwrap_or(0);
    let max_block_col = self.blocks.keys().map(|(_, c)| *c).max().unwrap_or(0);
    let grid_rows = max_block_row + 1;
    let grid_cols = max_block_col + 1;

    // Determine block sizes by examining existing blocks
    let mut row_block_sizes = vec![0; grid_rows];
    let mut col_block_sizes = vec![0; grid_cols];

    // Find row heights and column widths from existing blocks
    for ((block_row, block_col), matrix) in &self.blocks {
      row_block_sizes[*block_row] = row_block_sizes[*block_row].max(matrix.num_rows());
      col_block_sizes[*block_col] = col_block_sizes[*block_col].max(matrix.num_cols());
    }

    // If any block size is still 0, use a default size of 1
    for size in &mut row_block_sizes {
      if *size == 0 {
        *size = 1;
      }
    }
    for size in &mut col_block_sizes {
      if *size == 0 {
        *size = 1;
      }
    }

    // Calculate total dimensions
    let total_rows: usize = row_block_sizes.iter().sum();
    let total_cols: usize = col_block_sizes.iter().sum();

    // Create the full matrix by assembling blocks
    let mut full_matrix = Matrix::zeros(total_rows, total_cols);

    // Compute offsets for block placement
    let mut row_offsets = vec![0; grid_rows + 1];
    for i in 0..grid_rows {
      row_offsets[i + 1] = row_offsets[i] + row_block_sizes[i];
    }

    let mut col_offsets = vec![0; grid_cols + 1];
    for i in 0..grid_cols {
      col_offsets[i + 1] = col_offsets[i] + col_block_sizes[i];
    }

    // Place existing blocks
    for ((block_row, block_col), matrix) in &self.blocks {
      let row_start = row_offsets[*block_row];
      let col_start = col_offsets[*block_col];

      for i in 0..matrix.num_rows() {
        for j in 0..matrix.num_cols() {
          full_matrix.set(row_start + i, col_start + j, *matrix.get(i, j).unwrap());
        }
      }
    }

    // Now display the matrix with block boundaries
    self.display_matrix_with_blocks(f, &full_matrix, &row_block_sizes, &col_block_sizes)
  }
}

impl<F: Field + Copy + fmt::Display> BlockMatrixBuilder<F> {
  /// Display a matrix with block boundaries
  fn display_matrix_with_blocks(
    &self,
    f: &mut fmt::Formatter<'_>,
    matrix: &Matrix<F>,
    row_block_sizes: &[usize],
    col_block_sizes: &[usize],
  ) -> fmt::Result {
    let total_rows = matrix.num_rows();
    let total_cols = matrix.num_cols();

    if total_rows == 0 {
      return writeln!(f, "( )");
    }

    // Calculate column widths for proper alignment
    let mut col_widths = vec![0; total_cols];
    for i in 0..total_rows {
      #[allow(clippy::needless_range_loop)]
      for j in 0..total_cols {
        let element_str = format!("{}", matrix.get(i, j).unwrap());
        col_widths[j] = col_widths[j].max(element_str.len());
      }
    }

    // Compute block boundary positions
    let row_boundaries = self.compute_row_boundaries(row_block_sizes);
    let col_boundaries = self.compute_col_boundaries(col_block_sizes);

    for i in 0..total_rows {
      // Print horizontal separator before this row if it's a block boundary (but not the first row)
      if i > 0 && row_boundaries.contains(&i) {
        // Print left parenthesis part for separator row
        if total_rows == 1 {
          write!(f, "│ ")?; // Won't happen since single row has no separators
        } else {
          write!(f, "⎜ ")?; // Mid-section of parenthesis
        }

        self.print_horizontal_separator(f, &col_widths, &col_boundaries)?;

        // Print right parenthesis part for separator row
        writeln!(f, " ⎟")?;
      }

      // Print the row with vertical separators and parentheses
      // Left parenthesis
      if total_rows == 1 {
        write!(f, "( ")?; // Single row: simple parentheses
      } else if i == 0 {
        write!(f, "⎛ ")?; // Top of parenthesis
      } else if i == total_rows - 1 {
        write!(f, "⎝ ")?; // Bottom of parenthesis
      } else {
        write!(f, "⎜ ")?; // Middle of parenthesis
      }

      // Print matrix elements with block separators
      #[allow(clippy::needless_range_loop)]
      for j in 0..total_cols {
        if j > 0 {
          if col_boundaries.contains(&j) {
            write!(f, " │ ")?; // Unicode vertical line with padding for block separator
          } else {
            write!(f, "  ")?; // Regular spacing between elements in same block
          }
        }
        write!(f, "{:>width$}", matrix.get(i, j).unwrap(), width = col_widths[j])?;
      }

      // Right parenthesis
      if total_rows == 1 {
        writeln!(f, " )")?; // Single row: simple parentheses
      } else if i == 0 {
        writeln!(f, " ⎞")?; // Top of parenthesis
      } else if i == total_rows - 1 {
        writeln!(f, " ⎠")?; // Bottom of parenthesis
      } else {
        writeln!(f, " ⎟")?; // Middle of parenthesis
      }
    }

    Ok(())
  }

  /// Computes the row indices where block boundaries occur
  fn compute_row_boundaries(&self, row_block_sizes: &[usize]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut current_row = 0;

    for &block_height in row_block_sizes {
      current_row += block_height;
      boundaries.push(current_row);
    }

    // Remove the last boundary (which is the total number of rows)
    boundaries.pop();
    boundaries
  }

  /// Computes the column indices where block boundaries occur
  fn compute_col_boundaries(&self, col_block_sizes: &[usize]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut current_col = 0;

    for &block_width in col_block_sizes {
      current_col += block_width;
      boundaries.push(current_col);
    }

    // Remove the last boundary (which is the total number of columns)
    boundaries.pop();
    boundaries
  }

  /// Prints a horizontal separator line with proper alignment
  fn print_horizontal_separator(
    &self,
    f: &mut fmt::Formatter<'_>,
    col_widths: &[usize],
    col_boundaries: &[usize],
  ) -> fmt::Result {
    for (j, &width) in col_widths.iter().enumerate() {
      if j > 0 {
        if col_boundaries.contains(&j) {
          write!(f, "─┼─")?; // Unicode cross at block boundary
        } else {
          write!(f, "──")?; // Unicode horizontal line for regular spacing
        }
      }
      write!(f, "{}", "─".repeat(width))?; // Unicode horizontal line for column width
    }

    Ok(())
  }
}

// Matrix-vector multiplication
impl<F: Field + Copy> std::ops::Mul<Vector<F>> for Matrix<F> {
  type Output = Vector<F>;

  fn mul(self, rhs: Vector<F>) -> Self::Output {
    assert_eq!(
      self.num_cols(),
      rhs.dimension(),
      "Matrix-vector dimension mismatch: {}x{} * {}",
      self.num_rows(),
      self.num_cols(),
      rhs.dimension()
    );

    let components: Vec<F> = self
      .rows
      .iter()
      .map(|row| {
        row
          .components()
          .iter()
          .zip(rhs.components().iter())
          .map(|(&a, &b)| a * b)
          .fold(F::zero(), |acc, x| acc + x)
      })
      .collect();

    Vector::new(components)
  }
}

// Matrix-matrix multiplication
impl<F: Field + Copy> std::ops::Mul<Matrix<F>> for Matrix<F> {
  type Output = Matrix<F>;

  fn mul(self, rhs: Matrix<F>) -> Self::Output {
    assert_eq!(
      self.num_cols(),
      rhs.num_rows(),
      "Matrix dimension mismatch: {}x{} * {}x{}",
      self.num_rows(),
      self.num_cols(),
      rhs.num_rows(),
      rhs.num_cols()
    );

    let mut result = Matrix::zeros(self.num_rows(), rhs.num_cols());

    for i in 0..self.num_rows() {
      for j in 0..rhs.num_cols() {
        let mut sum = F::zero();
        for k in 0..self.num_cols() {
          sum += *self.get(i, k).unwrap() * *rhs.get(k, j).unwrap();
        }
        result.set(i, j, sum);
      }
    }

    result
  }
}

impl<F: Field + Copy + fmt::Display> fmt::Display for Matrix<F> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.is_empty() {
      return write!(f, "( )");
    }

    // Calculate column widths for alignment
    let mut col_widths = vec![0; self.num_cols()];
    for row in &self.rows {
      for (j, component) in row.components().iter().enumerate() {
        let element_str = format!("{component}");
        col_widths[j] = col_widths[j].max(element_str.len());
      }
    }

    // Format with mathematical parentheses
    for (i, row) in self.rows.iter().enumerate() {
      // Print appropriate parenthesis for this row
      if self.num_rows() == 1 {
        write!(f, "( ")?;
      } else if i == 0 {
        write!(f, "⎛ ")?;
      } else if i == self.num_rows() - 1 {
        write!(f, "⎝ ")?;
      } else {
        write!(f, "⎜ ")?;
      }

      // Print row elements
      for (j, component) in row.components().iter().enumerate() {
        if j > 0 {
          write!(f, "  ")?;
        }
        write!(f, "{:>width$}", component, width = col_widths[j])?;
      }

      // Print closing parenthesis
      if self.num_rows() == 1 {
        write!(f, " )")?;
      } else if i == 0 {
        writeln!(f, " ⎞")?;
      } else if i == self.num_rows() - 1 {
        write!(f, " ⎠")?;
      } else {
        writeln!(f, " ⎟")?;
      }
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_matrix_creation() {
    let matrix = Matrix::builder().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).build();

    assert_eq!(matrix.dimensions(), (2, 3));
    assert_eq!(matrix[(0, 1)], 2.0);
    assert_eq!(matrix[(1, 2)], 6.0);
  }

  #[test]
  fn test_from_rows() {
    let rows = vec![Vector::from([1.0, 2.0]), Vector::from([3.0, 4.0])];
    let matrix = Matrix::from_rows(rows);

    assert_eq!(matrix.dimensions(), (2, 2));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 1)], 4.0);
  }

  #[test]
  fn test_from_cols() {
    let cols = vec![Vector::from([1.0, 3.0]), Vector::from([2.0, 4.0])];
    let matrix = Matrix::from_cols(cols);

    assert_eq!(matrix.dimensions(), (2, 2));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(0, 1)], 2.0);
    assert_eq!(matrix[(1, 0)], 3.0);
    assert_eq!(matrix[(1, 1)], 4.0);
  }

  #[test]
  fn test_zeros_and_identity() {
    let zeros = Matrix::<f64>::zeros(2, 3);
    assert_eq!(zeros.dimensions(), (2, 3));
    assert_eq!(zeros[(0, 0)], 0.0);

    let identity = Matrix::<f64>::identity(3);
    assert_eq!(identity.dimensions(), (3, 3));
    assert_eq!(identity[(0, 0)], 1.0);
    assert_eq!(identity[(1, 1)], 1.0);
    assert_eq!(identity[(0, 1)], 0.0);
  }

  #[test]
  fn test_transpose() {
    let matrix = Matrix::builder().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).build();

    let transposed = matrix.transpose();
    assert_eq!(transposed.dimensions(), (3, 2));
    assert_eq!(transposed[(0, 0)], 1.0);
    assert_eq!(transposed[(1, 0)], 2.0);
    assert_eq!(transposed[(2, 1)], 6.0);
  }

  #[test]
  fn test_matrix_vector_multiplication() {
    let matrix = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    let vector = Vector::from([5.0, 6.0]);
    let result = matrix * vector;

    assert_eq!(result.components(), &[17.0, 39.0]); // [1*5+2*6, 3*5+4*6]
  }

  #[test]
  fn test_matrix_matrix_multiplication() {
    let a = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    let b = Matrix::builder().row([5.0, 6.0]).row([7.0, 8.0]).build();

    let result = a * b;
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result[(0, 0)], 19.0); // 1*5 + 2*7
    assert_eq!(result[(0, 1)], 22.0); // 1*6 + 2*8
    assert_eq!(result[(1, 0)], 43.0); // 3*5 + 4*7
    assert_eq!(result[(1, 1)], 50.0); // 3*6 + 4*8
  }

  #[test]
  fn test_row_echelon_form() {
    let matrix =
      Matrix::builder().row([1.0, 2.0, 3.0]).row([4.0, 5.0, 6.0]).row([7.0, 8.0, 9.0]).build();

    let (_rref, output) = matrix.into_row_echelon_form();
    assert_eq!(output.rank, 2);
    assert_eq!(output.pivots.len(), 2);
  }

  #[test]
  fn test_image_and_kernel() {
    let matrix = Matrix::builder().row([1.0_f64, 0.0, -1.0]).row([0.0, 1.0, 2.0]).build();

    let image = matrix.image();
    assert_eq!(image.len(), 2); // rank 2

    let kernel = matrix.kernel();
    assert_eq!(kernel.len(), 1); // nullity 1

    // Verify kernel vector satisfies Ax = 0
    if let Some(kernel_vec) = kernel.first() {
      let result = matrix * kernel_vec.clone();
      assert!(result.components().iter().all(|&x| x.abs() < 1e-10));
    }
  }

  #[test]
  fn test_display_formatting() {
    let matrix = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    let display_str = format!("{matrix}");
    assert!(display_str.contains("1"));
    assert!(display_str.contains("4"));
  }

  #[test]
  fn test_column_operations() {
    let mut matrix = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    let col = matrix.column(1);
    assert_eq!(col.components(), &[2.0, 4.0]);

    matrix.set_column(0, &Vector::from([5.0, 6.0]));
    assert_eq!(matrix[(0, 0)], 5.0);
    assert_eq!(matrix[(1, 0)], 6.0);
  }

  #[test]
  fn test_from_blocks() {
    // Test 2x2 block matrix with identity blocks on diagonal and zeros off-diagonal
    let blocks = vec![vec![Matrix::identity(2), Matrix::zeros(2, 3)], vec![
      Matrix::zeros(3, 2),
      Matrix::identity(3),
    ]];
    let block_matrix = Matrix::<f64>::from_blocks(blocks);

    assert_eq!(block_matrix.dimensions(), (5, 5));

    // Check identity block (0,0)
    assert_eq!(block_matrix[(0, 0)], 1.0);
    assert_eq!(block_matrix[(1, 1)], 1.0);
    assert_eq!(block_matrix[(0, 1)], 0.0);

    // Check zero block (0,1)
    assert_eq!(block_matrix[(0, 2)], 0.0);
    assert_eq!(block_matrix[(1, 3)], 0.0);

    // Check identity block (1,1)
    assert_eq!(block_matrix[(2, 2)], 1.0);
    assert_eq!(block_matrix[(3, 3)], 1.0);
    assert_eq!(block_matrix[(4, 4)], 1.0);
  }

  #[test]
  #[should_panic(expected = "Block at (0, 1) has 3 rows, expected 2")]
  fn test_from_blocks_mismatched_heights() {
    let blocks = vec![vec![Matrix::identity(2), Matrix::identity(3)]];
    let _block_matrix = Matrix::<f64>::from_blocks(blocks);
  }

  #[test]
  #[should_panic(expected = "Block at (1, 0) has 3 columns, expected 2")]
  fn test_from_blocks_mismatched_widths() {
    let blocks = vec![vec![Matrix::identity(2)], vec![Matrix::identity(3)]];
    let _block_matrix = Matrix::<f64>::from_blocks(blocks);
  }

  #[test]
  fn test_block_builder() {
    let block_matrix = Matrix::block_builder()
      .block(0, 0, Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build())
      .block(1, 1, Matrix::builder().row([5.0]).build())
      .build(vec![2, 1], vec![2, 1]);

    assert_eq!(block_matrix.dimensions(), (3, 3));
    assert_eq!(block_matrix[(0, 0)], 1.0);
    assert_eq!(block_matrix[(2, 2)], 5.0);

    // Check zero blocks
    assert_eq!(block_matrix[(0, 2)], 0.0);
    assert_eq!(block_matrix[(2, 0)], 0.0);
  }

  #[test]
  fn test_extract_block() {
    let matrix = Matrix::builder()
      .row([1.0, 2.0, 3.0, 4.0])
      .row([5.0, 6.0, 7.0, 8.0])
      .row([9.0, 10.0, 11.0, 12.0])
      .build();

    let block = matrix.extract_block(0, 2, 1, 3);
    assert_eq!(block.dimensions(), (2, 2));
    assert_eq!(block[(0, 0)], 2.0);
    assert_eq!(block[(0, 1)], 3.0);
    assert_eq!(block[(1, 0)], 6.0);
    assert_eq!(block[(1, 1)], 7.0);
  }

  #[test]
  fn test_set_block() {
    let mut matrix = Matrix::zeros(4, 4);
    let block = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    matrix.set_block(1, 1, &block);

    assert_eq!(matrix[(1, 1)], 1.0);
    assert_eq!(matrix[(1, 2)], 2.0);
    assert_eq!(matrix[(2, 1)], 3.0);
    assert_eq!(matrix[(2, 2)], 4.0);

    // Check that other elements remain zero
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(3, 3)], 0.0);
  }

  #[test]
  fn test_block_diagonal() {
    let blocks = vec![
      Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build(),
      Matrix::builder().row([5.0]).build(),
      Matrix::identity(2),
    ];

    let block_diag = Matrix::block_diagonal(blocks);
    assert_eq!(block_diag.dimensions(), (5, 5));

    // First block
    assert_eq!(block_diag[(0, 0)], 1.0);
    assert_eq!(block_diag[(0, 1)], 2.0);
    assert_eq!(block_diag[(1, 0)], 3.0);
    assert_eq!(block_diag[(1, 1)], 4.0);

    // Second block
    assert_eq!(block_diag[(2, 2)], 5.0);

    // Third block (identity)
    assert_eq!(block_diag[(3, 3)], 1.0);
    assert_eq!(block_diag[(4, 4)], 1.0);
    assert_eq!(block_diag[(3, 4)], 0.0);

    // Check zeros in off-diagonal blocks
    assert_eq!(block_diag[(0, 2)], 0.0);
    assert_eq!(block_diag[(2, 0)], 0.0);
  }

  #[test]
  fn test_hstack() {
    let a = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();
    let b = Matrix::builder().row([5.0]).row([6.0]).build();
    let c = Matrix::builder().row([7.0, 8.0]).row([9.0, 10.0]).build();

    let result = Matrix::hstack([a, b, c]);
    assert_eq!(result.dimensions(), (2, 5));

    assert_eq!(result[(0, 0)], 1.0);
    assert_eq!(result[(0, 1)], 2.0);
    assert_eq!(result[(0, 2)], 5.0);
    assert_eq!(result[(0, 3)], 7.0);
    assert_eq!(result[(0, 4)], 8.0);

    assert_eq!(result[(1, 0)], 3.0);
    assert_eq!(result[(1, 1)], 4.0);
    assert_eq!(result[(1, 2)], 6.0);
    assert_eq!(result[(1, 3)], 9.0);
    assert_eq!(result[(1, 4)], 10.0);
  }

  #[test]
  fn test_vstack() {
    let a = Matrix::builder().row([1.0, 2.0, 3.0]).build();
    let b = Matrix::builder().row([4.0, 5.0, 6.0]).row([7.0, 8.0, 9.0]).build();

    let result = Matrix::vstack([a, b]);
    assert_eq!(result.dimensions(), (3, 3));

    assert_eq!(result[(0, 0)], 1.0);
    assert_eq!(result[(0, 1)], 2.0);
    assert_eq!(result[(0, 2)], 3.0);

    assert_eq!(result[(1, 0)], 4.0);
    assert_eq!(result[(1, 1)], 5.0);
    assert_eq!(result[(1, 2)], 6.0);

    assert_eq!(result[(2, 0)], 7.0);
    assert_eq!(result[(2, 1)], 8.0);
    assert_eq!(result[(2, 2)], 9.0);
  }

  #[test]
  #[should_panic(expected = "Matrix 1 has 1 rows, expected 2")]
  fn test_hstack_mismatched_rows() {
    let a = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();
    let b = Matrix::builder().row([5.0]).build();

    let _result = Matrix::hstack([a, b]);
  }

  #[test]
  #[should_panic(expected = "Matrix 1 has 1 columns, expected 2")]
  fn test_vstack_mismatched_cols() {
    let a = Matrix::builder().row([1.0, 2.0]).build();
    let b = Matrix::builder().row([3.0]).build();

    let _result = Matrix::vstack([a, b]);
  }

  #[test]
  fn test_empty_block_operations() {
    let empty_blocks: Vec<Matrix<f64>> = vec![];
    let empty_diag = Matrix::block_diagonal(empty_blocks);
    assert!(empty_diag.is_empty());

    let empty_hstack = Matrix::hstack(Vec::<Matrix<f64>>::new());
    assert!(empty_hstack.is_empty());

    let empty_vstack = Matrix::vstack(Vec::<Matrix<f64>>::new());
    assert!(empty_vstack.is_empty());
  }

  #[test]
  fn test_block_display() {
    // Test the BlockMatrixBuilder display functionality
    let builder = Matrix::<f64>::block_builder().block(0, 0, Matrix::<f64>::identity(2)).block(
      1,
      1,
      Matrix::<f64>::identity(3),
    );

    let display_str = format!("{builder}");

    // Verify it contains the expected structure - should show actual matrix with block boundaries
    assert!(display_str.contains("1"));
    assert!(display_str.contains("0"));
    assert!(display_str.contains("│")); // Block separator

    println!("Block matrix builder display:\n{display_str}");
  }

  #[test]
  fn test_block_builder_display() {
    // Create a block matrix using the builder and test display during construction
    let builder = Matrix::block_builder()
      .block(0, 0, Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build())
      .block(1, 1, Matrix::builder().row([5.0]).build());

    // Display the builder state
    println!("Block builder during construction:\n{builder}");

    // Build the final matrix
    let block_matrix = builder.build(vec![2, 1], vec![2, 1]);

    // Verify structure
    assert_eq!(block_matrix.dimensions(), (3, 3));
    assert_eq!(block_matrix[(0, 0)], 1.0);
    assert_eq!(block_matrix[(2, 2)], 5.0);
  }

  #[test]
  fn test_matrix_builder_columns() {
    // Test building matrix using column methods
    let matrix = Matrix::builder().column([1.0, 3.0, 5.0]).column([2.0, 4.0, 6.0]).build();

    assert_eq!(matrix.dimensions(), (3, 2));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(0, 1)], 2.0);
    assert_eq!(matrix[(1, 0)], 3.0);
    assert_eq!(matrix[(1, 1)], 4.0);
    assert_eq!(matrix[(2, 0)], 5.0);
    assert_eq!(matrix[(2, 1)], 6.0);
  }

  #[test]
  fn test_matrix_builder_column_vec() {
    // Test building matrix using column_vec method
    let col1 = Vector::from([1.0, 3.0]);
    let col2 = Vector::from([2.0, 4.0]);

    let matrix = Matrix::builder().column_vec(col1).column_vec(col2).build();

    assert_eq!(matrix.dimensions(), (2, 2));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(0, 1)], 2.0);
    assert_eq!(matrix[(1, 0)], 3.0);
    assert_eq!(matrix[(1, 1)], 4.0);
  }

  #[test]
  fn test_matrix_builder_column_iter() {
    // Test building matrix using column_iter method
    let matrix = Matrix::builder()
      .column_iter([1.0, 4.0, 7.0])
      .column_iter([2.0, 5.0, 8.0])
      .column_iter([3.0, 6.0, 9.0])
      .build();

    assert_eq!(matrix.dimensions(), (3, 3));
    // First column: [1, 4, 7]
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 4.0);
    assert_eq!(matrix[(2, 0)], 7.0);
    // Second column: [2, 5, 8]
    assert_eq!(matrix[(0, 1)], 2.0);
    assert_eq!(matrix[(1, 1)], 5.0);
    assert_eq!(matrix[(2, 1)], 8.0);
    // Third column: [3, 6, 9]
    assert_eq!(matrix[(0, 2)], 3.0);
    assert_eq!(matrix[(1, 2)], 6.0);
    assert_eq!(matrix[(2, 2)], 9.0);
  }

  #[test]
  #[should_panic(expected = "Cannot mix row and column operations in MatrixBuilder")]
  fn test_matrix_builder_mixed_operations_panic() {
    // Test that mixing row and column operations panics
    let _matrix = Matrix::builder()
      .row([1.0, 2.0])
      .column([3.0, 4.0]) // This should panic
      .build();
  }
}
