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
//! let zeros = Matrix::zeros(3, 3);
//! let identity = Matrix::identity(3);
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
//! let block_matrix =
//!   Matrix::from_blocks([[Some(Matrix::identity(2)), None], [None, Some(Matrix::identity(3))]]);
//!
//! // Or using block builder
//! let block_matrix = Matrix::block_builder()
//!   .block(0, 0, Matrix::identity(2))
//!   .block(1, 1, Matrix::identity(3))
//!   .build([2, 3], [2, 3]);
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

  /// Creates a matrix from a 2D array of optional blocks.
  ///
  /// Each `Some(matrix)` represents a non-zero block, while `None` represents a zero block.
  /// All blocks in the same block-row must have the same height, and all blocks in the
  /// same block-column must have the same width.
  ///
  /// # Examples
  ///
  /// ```
  /// use cova_algebra::tensors::dynamic::matrix::Matrix;
  ///
  /// // Create a 2x2 block matrix with identity blocks on the diagonal
  /// let block_matrix =
  ///   Matrix::from_blocks([[Some(Matrix::identity(2)), None], [None, Some(Matrix::identity(3))]]);
  /// ```
  pub fn from_blocks<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
    blocks: [[Option<Matrix<F>>; BLOCK_COLS]; BLOCK_ROWS],
  ) -> Self {
    if BLOCK_ROWS == 0 || BLOCK_COLS == 0 {
      return Self::new();
    }

    // Determine block sizes by examining the first non-None block in each row/column
    let mut row_block_sizes = vec![0; BLOCK_ROWS];
    let mut col_block_sizes = vec![0; BLOCK_COLS];

    // Find row heights
    for (block_row, row) in blocks.iter().enumerate() {
      for block in row.iter().flatten() {
        if row_block_sizes[block_row] == 0 {
          row_block_sizes[block_row] = block.num_rows();
        } else {
          assert_eq!(
            block.num_rows(),
            row_block_sizes[block_row],
            "All blocks in block-row {} must have the same height",
            block_row
          );
        }
      }
    }

    // Find column widths
    for block_col in 0..BLOCK_COLS {
      for block_row in 0..BLOCK_ROWS {
        if let Some(ref block) = blocks[block_row][block_col] {
          if col_block_sizes[block_col] == 0 {
            col_block_sizes[block_col] = block.num_cols();
          } else {
            assert_eq!(
              block.num_cols(),
              col_block_sizes[block_col],
              "All blocks in block-column {} must have the same width",
              block_col
            );
          }
        }
      }
    }

    // Validate that we found sizes for all blocks
    for (i, &size) in row_block_sizes.iter().enumerate() {
      assert!(size > 0, "Block row {} has no non-zero blocks to determine size", i);
    }
    for (i, &size) in col_block_sizes.iter().enumerate() {
      assert!(size > 0, "Block column {} has no non-zero blocks to determine size", i);
    }

    // Calculate total dimensions
    let total_rows: usize = row_block_sizes.iter().sum();
    let total_cols: usize = col_block_sizes.iter().sum();

    let mut result = Self::zeros(total_rows, total_cols);

    // Compute offsets for efficient placement
    let mut row_offsets = vec![0; BLOCK_ROWS + 1];
    for i in 0..BLOCK_ROWS {
      row_offsets[i + 1] = row_offsets[i] + row_block_sizes[i];
    }

    let mut col_offsets = vec![0; BLOCK_COLS + 1];
    for i in 0..BLOCK_COLS {
      col_offsets[i + 1] = col_offsets[i] + col_block_sizes[i];
    }

    // Place blocks
    for (block_row, row) in blocks.iter().enumerate() {
      for (block_col, block_opt) in row.iter().enumerate() {
        if let Some(ref block) = block_opt {
          let row_start = row_offsets[block_row];
          let col_start = col_offsets[block_col];

          for i in 0..block.num_rows() {
            for j in 0..block.num_cols() {
              result.set(row_start + i, col_start + j, *block.get(i, j).unwrap());
            }
          }
        }
      }
    }

    result
  }

  /// Sets the component at the given row and column.
  pub fn set(&mut self, row: usize, col: usize, value: F) {
    assert!(row < self.num_rows(), "Row index {} out of bounds", row);
    assert!(col < self.num_cols(), "Column index {} out of bounds", col);
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
    assert!(index < self.num_cols(), "Column index {} out of bounds", index);

    let components: Vec<F> = self.rows.iter().map(|row| *row.get_component(index)).collect();

    Vector::new(components)
  }

  /// Sets the column at the given index.
  pub fn set_column(&mut self, index: usize, column: &Vector<F>) {
    assert!(index < self.num_cols(), "Column index {} out of bounds", index);
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
  rows: Vec<Vector<F>>,
}

impl<F: Field + Copy> MatrixBuilder<F> {
  /// Creates a new matrix builder.
  pub fn new() -> Self { Self { rows: Vec::new() } }

  /// Adds a row from an array.
  pub fn row<const N: usize>(mut self, row: [F; N]) -> Self {
    self.rows.push(Vector::from(row));
    self
  }

  /// Adds a row from a vector.
  pub fn row_vec(mut self, row: Vector<F>) -> Self {
    self.rows.push(row);
    self
  }

  /// Adds a row from an iterator.
  pub fn row_iter<I>(mut self, row: I) -> Self
  where I: IntoIterator<Item = F> {
    self.rows.push(Vector::from(row.into_iter().collect::<Vec<_>>()));
    self
  }

  /// Builds the matrix.
  pub fn build(self) -> Matrix<F> { Matrix::from_rows(self.rows) }
}

impl<F: Field + Copy> Default for MatrixBuilder<F> {
  fn default() -> Self { Self::new() }
}

/// Builder for constructing block matrices with a fluent API.
///
/// This builder allows you to specify blocks at specific positions and then
/// build the final matrix with the specified block structure.
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
  /// * `row_block_sizes`: Array specifying the height of each block row
  /// * `col_block_sizes`: Array specifying the width of each block column
  ///
  /// # Panics
  ///
  /// Panics if any block has dimensions that don't match the specified structure.
  pub fn build<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
    self,
    row_block_sizes: [usize; BLOCK_ROWS],
    col_block_sizes: [usize; BLOCK_COLS],
  ) -> Matrix<F> {
    // Validate block dimensions
    for ((block_row, block_col), matrix) in &self.blocks {
      assert!(
        *block_row < BLOCK_ROWS,
        "Block row {} out of bounds (max: {})",
        block_row,
        BLOCK_ROWS - 1
      );
      assert!(
        *block_col < BLOCK_COLS,
        "Block column {} out of bounds (max: {})",
        block_col,
        BLOCK_COLS - 1
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
    let mut row_offsets = vec![0; BLOCK_ROWS + 1];
    for i in 0..BLOCK_ROWS {
      row_offsets[i + 1] = row_offsets[i] + row_block_sizes[i];
    }

    let mut col_offsets = vec![0; BLOCK_COLS + 1];
    for i in 0..BLOCK_COLS {
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

impl<F: Field + Copy> Matrix<F> {
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
  /// let blocks = vec![Matrix::identity(2), Matrix::identity(3)];
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
        let element_str = format!("{}", component);
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
  use crate::fixtures::Mod7;

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

    let (rref, output) = matrix.into_row_echelon_form();
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
      let result = matrix.clone() * kernel_vec.clone();
      assert!(result.components().iter().all(|&x| x.abs() < 1e-10));
    }
  }

  #[test]
  fn test_display_formatting() {
    let matrix = Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build();

    let display_str = format!("{}", matrix);
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
    // Test 2x2 block matrix with identity blocks on diagonal
    let block_matrix = Matrix::<f64>::from_blocks([[Some(Matrix::identity(2)), None], [
      None,
      Some(Matrix::identity(3)),
    ]]);

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
  fn test_block_builder() {
    let block_matrix = Matrix::block_builder()
      .block(0, 0, Matrix::builder().row([1.0, 2.0]).row([3.0, 4.0]).build())
      .block(1, 1, Matrix::builder().row([5.0]).build())
      .build([2, 1], [2, 1]);

    assert_eq!(block_matrix.dimensions(), (3, 3));
    assert_eq!(block_matrix[(0, 0)], 1.0);
    assert_eq!(block_matrix[(0, 1)], 2.0);
    assert_eq!(block_matrix[(1, 0)], 3.0);
    assert_eq!(block_matrix[(1, 1)], 4.0);
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
  #[should_panic(expected = "All blocks in block-row 0 must have the same height")]
  fn test_from_blocks_mismatched_heights() {
    let _block_matrix =
      Matrix::<f64>::from_blocks([[Some(Matrix::identity(2)), Some(Matrix::identity(3))]]);
  }

  #[test]
  #[should_panic(expected = "All blocks in block-column 0 must have the same width")]
  fn test_from_blocks_mismatched_widths() {
    let _block_matrix =
      Matrix::<f64>::from_blocks([[Some(Matrix::identity(2))], [Some(Matrix::identity(3))]]);
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
}
