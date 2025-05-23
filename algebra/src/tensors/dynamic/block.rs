//! # Block Matrix Module
//!
//! This module provides efficient block matrix implementations for linear algebra operations
//! where matrices have a natural block structure. Block matrices are particularly useful
//! in applications like cellular sheaf cohomology, finite element methods, and structured
//! linear systems.
//!
//! ## Core Concepts
//!
//! A block matrix is a matrix that can be partitioned into rectangular blocks:
//!
//! ```text
//! M = [ A11  A12  A13 ]
//!     [ A21  A22  A23 ]
//!     [ A31  A32  A33 ]
//! ```
//!
//! Where each Aᵢⱼ is itself a matrix (possibly zero). This implementation stores only
//! non-zero blocks for efficiency.
//!
//! ## Usage
//!
//! ```
//! use harness_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{
//!     block::BlockMatrix,
//!     matrix::{DynamicDenseMatrix, RowMajor},
//!     vector::DynamicVector,
//!   },
//! };
//!
//! // Create a 2x2 block matrix where each block has specified dimensions
//! let row_sizes = vec![2, 3]; // First block row has height 2, second has height 3
//! let col_sizes = vec![2, 1]; // First block col has width 2, second has width 1
//! let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(row_sizes, col_sizes);
//!
//! // Set a block at position (0, 0)
//! let mut block_00 = DynamicDenseMatrix::<f64, RowMajor>::new();
//! block_00.append_row(DynamicVector::from([1.0, 2.0]));
//! block_00.append_row(DynamicVector::from([3.0, 4.0]));
//! block_matrix.set_block(0, 0, block_00);
//!
//! // Convert to a regular matrix when needed
//! let flat_matrix = block_matrix.flatten();
//! ```

use std::{collections::HashMap, fmt};

use super::matrix::{DynamicDenseMatrix, MatrixOrientation, RowMajor};
use crate::prelude::*;

/// A sparse block matrix that stores only non-zero blocks.
///
/// The block matrix maintains a regular block structure where all blocks in the same
/// block-row have the same height, and all blocks in the same block-column have the
/// same width. Zero blocks are not stored, providing memory efficiency for sparse
/// block structures.
///
/// # Type Parameters
///
/// * `F`: The field type for matrix elements, must implement [`Field`] and `Copy`
/// * `O`: The matrix orientation, either `RowMajor` or `ColumnMajor`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMatrix<F, O>
where
  F: Field + Copy,
  O: MatrixOrientation, {
  /// Stores only non-zero blocks, indexed by (block_row, block_col)
  blocks:          HashMap<(usize, usize), DynamicDenseMatrix<F, O>>,
  /// Number of block rows in the matrix
  num_block_rows:  usize,
  /// Number of block columns in the matrix
  num_block_cols:  usize,
  /// Height of each block row (number of scalar rows in each block row)
  row_block_sizes: Vec<usize>,
  /// Width of each block column (number of scalar columns in each block column)
  col_block_sizes: Vec<usize>,
}

impl<F> BlockMatrix<F, RowMajor>
where F: Field + Copy
{
  /// Creates a new empty block matrix with the specified block structure.
  ///
  /// # Arguments
  ///
  /// * `row_block_sizes`: Vector specifying the height (number of rows) of each block row
  /// * `col_block_sizes`: Vector specifying the width (number of columns) of each block column
  ///
  /// # Panics
  ///
  /// Panics if either `row_block_sizes` or `col_block_sizes` is empty.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::tensors::dynamic::{
  ///   block::BlockMatrix,
  ///   matrix::{DynamicDenseMatrix, RowMajor},
  /// };
  ///
  /// // Create a 2x3 block matrix structure
  /// let row_sizes = vec![2, 3]; // Two block rows: heights 2 and 3
  /// let col_sizes = vec![1, 2, 1]; // Three block columns: widths 1, 2, and 1
  /// let block_matrix = BlockMatrix::<f64, RowMajor>::new(row_sizes, col_sizes);
  /// ```
  pub fn new(row_block_sizes: Vec<usize>, col_block_sizes: Vec<usize>) -> Self {
    assert!(row_block_sizes.iter().all(|&size| size > 0), "All block row sizes must be positive");
    assert!(
      col_block_sizes.iter().all(|&size| size > 0),
      "All block column sizes must be positive"
    );

    let num_block_rows = row_block_sizes.len();
    let num_block_cols = col_block_sizes.len();

    Self {
      blocks: HashMap::new(),
      num_block_rows,
      num_block_cols,
      row_block_sizes,
      col_block_sizes,
    }
  }

  /// Creates a new block matrix filled with zero blocks of the specified structure.
  ///
  /// This is equivalent to `new()` but makes the intent clearer when you want to
  /// emphasize that the matrix starts as all zeros.
  pub fn zeros(row_block_sizes: Vec<usize>, col_block_sizes: Vec<usize>) -> Self {
    Self::new(row_block_sizes, col_block_sizes)
  }

  /// Sets a block at the specified position.
  ///
  /// The block dimensions must exactly match the expected dimensions for that position
  /// based on the block structure defined during construction.
  ///
  /// # Arguments
  ///
  /// * `block_row`: The block row index (0-based)
  /// * `block_col`: The block column index (0-based)
  /// * `block`: The matrix block to place at this position
  ///
  /// # Panics
  ///
  /// * Panics if `block_row` or `block_col` are out of bounds
  /// * Panics if the block dimensions don't match the expected dimensions for this position
  pub fn set_block(
    &mut self,
    block_row: usize,
    block_col: usize,
    block: DynamicDenseMatrix<F, RowMajor>,
  ) {
    assert!(
      block_row < self.num_block_rows,
      "Block row index {} out of bounds (max: {})",
      block_row,
      self.num_block_rows - 1
    );
    assert!(
      block_col < self.num_block_cols,
      "Block column index {} out of bounds (max: {})",
      block_col,
      self.num_block_cols - 1
    );

    let expected_rows = self.row_block_sizes[block_row];
    let expected_cols = self.col_block_sizes[block_col];

    assert_eq!(
      block.num_rows(),
      expected_rows,
      "Block has {} rows but position ({}, {}) requires {} rows",
      block.num_rows(),
      block_row,
      block_col,
      expected_rows
    );
    assert_eq!(
      block.num_cols(),
      expected_cols,
      "Block has {} columns but position ({}, {}) requires {} columns",
      block.num_cols(),
      block_row,
      block_col,
      expected_cols
    );

    if is_effectively_zero(&block) {
      self.blocks.remove(&(block_row, block_col));
    } else {
      self.blocks.insert((block_row, block_col), block);
    }
  }

  /// Gets a reference to the block at the specified position.
  ///
  /// Returns `None` if the block is zero (not stored) or if the indices are out of bounds.
  pub fn get_block(
    &self,
    block_row: usize,
    block_col: usize,
  ) -> Option<&DynamicDenseMatrix<F, RowMajor>> {
    if block_row >= self.num_block_rows || block_col >= self.num_block_cols {
      return None;
    }
    self.blocks.get(&(block_row, block_col))
  }

  /// Gets a block at the specified position, returning a zero matrix if the block doesn't exist.
  ///
  /// # Panics
  ///
  /// Panics if `block_row` or `block_col` are out of bounds.
  pub fn get_block_or_zero(
    &self,
    block_row: usize,
    block_col: usize,
  ) -> DynamicDenseMatrix<F, RowMajor> {
    assert!(block_row < self.num_block_rows, "Block row index {block_row} out of bounds");
    assert!(block_col < self.num_block_cols, "Block column index {block_col} out of bounds");

    self.get_block(block_row, block_col).map_or_else(
      || {
        DynamicDenseMatrix::<F, RowMajor>::zeros(
          self.row_block_sizes[block_row],
          self.col_block_sizes[block_col],
        )
      },
      std::clone::Clone::clone,
    )
  }

  /// Checks if a block exists (is non-zero) at the specified position.
  pub fn has_block(&self, block_row: usize, block_col: usize) -> bool {
    self.blocks.contains_key(&(block_row, block_col))
  }

  /// Converts the block matrix to a flat `DynamicDenseMatrix`.
  ///
  /// This creates a new matrix by assembling all the blocks into a single
  /// contiguous matrix structure.
  pub fn flatten(&self) -> DynamicDenseMatrix<F, RowMajor> {
    let total_rows: usize = self.row_block_sizes.iter().sum();
    let total_cols: usize = self.col_block_sizes.iter().sum();

    let mut result = DynamicDenseMatrix::<F, RowMajor>::zeros(total_rows, total_cols);

    let row_offsets = self.compute_row_offsets();
    let col_offsets = self.compute_col_offsets();

    for ((block_row, block_col), block) in &self.blocks {
      let row_start = row_offsets[*block_row];
      let col_start = col_offsets[*block_col];

      for i in 0..block.num_rows() {
        for j in 0..block.num_cols() {
          result.set_component(row_start + i, col_start + j, *block.get_component(i, j));
        }
      }
    }

    result
  }

  /// Returns the block structure dimensions as (num_block_rows, num_block_cols).
  pub const fn block_structure(&self) -> (usize, usize) {
    (self.num_block_rows, self.num_block_cols)
  }

  /// Returns a slice of the row block sizes.
  pub fn row_block_sizes(&self) -> &[usize] { &self.row_block_sizes }

  /// Returns a slice of the column block sizes.
  pub fn col_block_sizes(&self) -> &[usize] { &self.col_block_sizes }

  /// Returns the total dimensions of the flattened matrix as (total_rows, total_cols).
  pub fn total_dimensions(&self) -> (usize, usize) {
    (self.row_block_sizes.iter().sum(), self.col_block_sizes.iter().sum())
  }

  /// Returns the number of non-zero blocks currently stored.
  pub fn num_nonzero_blocks(&self) -> usize { self.blocks.len() }

  /// Returns an iterator over all non-zero blocks as ((block_row, block_col), &block).
  pub fn iter_blocks(
    &self,
  ) -> impl Iterator<Item = ((usize, usize), &DynamicDenseMatrix<F, RowMajor>)> {
    self.blocks.iter().map(|((r, c), block)| ((*r, *c), block))
  }

  /// Computes cumulative row offsets for efficient block placement.
  fn compute_row_offsets(&self) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(self.num_block_rows + 1);
    offsets.push(0);
    for &size in &self.row_block_sizes {
      offsets.push(offsets.last().unwrap() + size);
    }
    offsets
  }

  /// Computes cumulative column offsets for efficient block placement.
  fn compute_col_offsets(&self) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(self.num_block_cols + 1);
    offsets.push(0);
    for &size in &self.col_block_sizes {
      offsets.push(offsets.last().unwrap() + size);
    }
    offsets
  }
}

/// Checks if a matrix is effectively zero (all elements are zero).
fn is_effectively_zero<F>(matrix: &DynamicDenseMatrix<F, RowMajor>) -> bool
where F: Field + Copy {
  for i in 0..matrix.num_rows() {
    for j in 0..matrix.num_cols() {
      if !matrix.get_component(i, j).is_zero() {
        return false;
      }
    }
  }
  true
}

impl<F: Field + Copy + fmt::Display> fmt::Display for BlockMatrix<F, RowMajor> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let flattened = self.flatten();
    let total_rows = flattened.num_rows();
    let total_cols = flattened.num_cols();

    if total_rows == 0 {
      return writeln!(f, "  ( )");
    }

    // Calculate column widths for proper alignment
    let mut col_widths = vec![0; total_cols];
    for i in 0..total_rows {
      (0..total_cols).for_each(|j| {
        let element_str = format!("{}", flattened.get_component(i, j));
        col_widths[j] = col_widths[j].max(element_str.len());
      });
    }

    // Compute block boundary positions
    let row_boundaries = self.compute_row_boundaries();
    let col_boundaries = self.compute_col_boundaries();

    for i in 0..total_rows {
      // Print horizontal separator before this row if it's a block boundary (but not the first row)
      if i > 0 && row_boundaries.contains(&i) {
        // Print left parenthesis part for separator row
        if total_rows == 1 {
          write!(f, "  │ ")?; // Won't happen since single row has no separators
        } else {
          write!(f, "  ⎜ ")?; // Mid-section of parenthesis
        }

        print_horizontal_separator(f, &col_widths, &col_boundaries)?;

        // Print right parenthesis part for separator row
        writeln!(f, " ⎟")?;
      }

      // Print the row with vertical separators and parentheses
      // Left parenthesis
      if total_rows == 1 {
        write!(f, "  ( ")?; // Single row: simple parentheses
      } else if i == 0 {
        write!(f, "  ⎛ ")?; // Top of parenthesis
      } else if i == total_rows - 1 {
        write!(f, "  ⎝ ")?; // Bottom of parenthesis
      } else {
        write!(f, "  ⎜ ")?; // Middle of parenthesis
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
        write!(f, "{:>width$}", flattened.get_component(i, j), width = col_widths[j])?;
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
}

impl<F: Field + Copy + fmt::Display> BlockMatrix<F, RowMajor> {
  /// Computes the row indices where block boundaries occur
  fn compute_row_boundaries(&self) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut current_row = 0;

    for &block_height in &self.row_block_sizes {
      current_row += block_height;
      boundaries.push(current_row);
    }

    // Remove the last boundary (which is the total number of rows)
    boundaries.pop();
    boundaries
  }

  /// Computes the column indices where block boundaries occur
  fn compute_col_boundaries(&self) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut current_col = 0;

    for &block_width in &self.col_block_sizes {
      current_col += block_width;
      boundaries.push(current_col);
    }

    // Remove the last boundary (which is the total number of columns)
    boundaries.pop();
    boundaries
  }
}

/// Prints a horizontal separator line with proper alignment
fn print_horizontal_separator(
  f: &mut fmt::Formatter<'_>,
  col_widths: &[usize],
  col_boundaries: &[usize],
) -> fmt::Result {
  // TODO (autoparallel): This is surely a bug right? It gets used in the loop below.
  #[allow(unused_variables)]
  let mut total_written = 0;

  for (j, &width) in col_widths.iter().enumerate() {
    if j > 0 {
      if col_boundaries.contains(&j) {
        write!(f, "─┼─")?; // Unicode cross at block boundary
        total_written += 3;
      } else {
        write!(f, "──")?; // Unicode horizontal line for regular spacing
        total_written += 2;
      }
    }
    write!(f, "{}", "─".repeat(width))?; // Unicode horizontal line for column width
    total_written += width;
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  #![allow(clippy::float_cmp)]
  use super::*;
  use crate::tensors::dynamic::{matrix::RowMajor, vector::DynamicVector};

  #[test]
  fn test_block_matrix_construction() {
    let row_sizes = vec![2, 3];
    let col_sizes = vec![1, 2];
    let block_matrix = BlockMatrix::<f64, RowMajor>::new(row_sizes, col_sizes);

    assert_eq!(block_matrix.block_structure(), (2, 2));
    assert_eq!(block_matrix.row_block_sizes(), &[2, 3]);
    assert_eq!(block_matrix.col_block_sizes(), &[1, 2]);
    assert_eq!(block_matrix.total_dimensions(), (5, 3));
    assert_eq!(block_matrix.num_nonzero_blocks(), 0);
  }

  #[test]
  fn test_set_and_get_block() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2, 1], vec![2, 1]);

    let mut block = DynamicDenseMatrix::<f64, RowMajor>::new();
    block.append_row(DynamicVector::from([1.0, 2.0]));
    block.append_row(DynamicVector::from([3.0, 4.0]));

    assert!(!block_matrix.has_block(0, 0));
    block_matrix.set_block(0, 0, block.clone());
    assert!(block_matrix.has_block(0, 0));

    let retrieved = block_matrix.get_block(0, 0).unwrap();
    assert_eq!(*retrieved.get_component(0, 0), 1.0);
    assert_eq!(*retrieved.get_component(0, 1), 2.0);
    assert_eq!(*retrieved.get_component(1, 0), 3.0);
    assert_eq!(*retrieved.get_component(1, 1), 4.0);
  }

  #[test]
  fn test_get_block_or_zero() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);

    let zero_block = block_matrix.get_block_or_zero(0, 0);
    assert_eq!(zero_block.num_rows(), 2);
    assert_eq!(zero_block.num_cols(), 2);
    assert_eq!(*zero_block.get_component(0, 0), 0.0);

    let mut actual_block = DynamicDenseMatrix::<f64, RowMajor>::new();
    actual_block.append_row(DynamicVector::from([1.0, 2.0]));
    actual_block.append_row(DynamicVector::from([3.0, 4.0]));
    block_matrix.set_block(0, 0, actual_block);

    let retrieved_block = block_matrix.get_block_or_zero(0, 0);
    assert_eq!(*retrieved_block.get_component(0, 0), 1.0);
    assert_eq!(*retrieved_block.get_component(1, 1), 4.0);
  }

  #[test]
  #[should_panic(expected = "Block row index 1 out of bounds")]
  fn test_set_block_row_out_of_bounds() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);
    let block = DynamicDenseMatrix::<f64, RowMajor>::zeros(2, 2);
    block_matrix.set_block(1, 0, block);
  }

  #[test]
  #[should_panic(expected = "Block column index 1 out of bounds")]
  fn test_set_block_col_out_of_bounds() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);
    let block = DynamicDenseMatrix::<f64, RowMajor>::zeros(2, 2);
    block_matrix.set_block(0, 1, block);
  }

  #[test]
  #[should_panic(expected = "Block has 3 rows but position (0, 0) requires 2 rows")]
  fn test_wrong_block_size_rows() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);
    let mut block = DynamicDenseMatrix::<f64, RowMajor>::new();
    block.append_row(DynamicVector::from([1.0, 2.0]));
    block.append_row(DynamicVector::from([3.0, 4.0]));
    block.append_row(DynamicVector::from([5.0, 6.0]));
    block_matrix.set_block(0, 0, block);
  }

  #[test]
  #[should_panic(expected = "Block has 3 columns but position (0, 0) requires 2 columns")]
  fn test_wrong_block_size_cols() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);
    let mut block = DynamicDenseMatrix::<f64, RowMajor>::new();
    block.append_row(DynamicVector::from([1.0, 2.0, 3.0]));
    block.append_row(DynamicVector::from([4.0, 5.0, 6.0]));
    block_matrix.set_block(0, 0, block);
  }

  #[test]
  fn test_zero_block_not_stored() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);
    let zero_block = DynamicDenseMatrix::<f64, RowMajor>::zeros(2, 2);

    block_matrix.set_block(0, 0, zero_block);
    assert!(!block_matrix.has_block(0, 0));
    assert_eq!(block_matrix.num_nonzero_blocks(), 0);
  }

  #[test]
  fn test_flatten_simple() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2], vec![2]);

    let mut block = DynamicDenseMatrix::<f64, RowMajor>::new();
    block.append_row(DynamicVector::from([1.0, 2.0]));
    block.append_row(DynamicVector::from([3.0, 4.0]));
    block_matrix.set_block(0, 0, block);

    let flat = block_matrix.flatten();
    assert_eq!(flat.num_rows(), 2);
    assert_eq!(flat.num_cols(), 2);
    assert_eq!(*flat.get_component(0, 0), 1.0);
    assert_eq!(*flat.get_component(0, 1), 2.0);
    assert_eq!(*flat.get_component(1, 0), 3.0);
    assert_eq!(*flat.get_component(1, 1), 4.0);
  }

  #[test]
  fn test_flatten_complex() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2, 1], vec![2, 1]);

    let mut block_00 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_00.append_row(DynamicVector::from([1.0, 2.0]));
    block_00.append_row(DynamicVector::from([3.0, 4.0]));
    block_matrix.set_block(0, 0, block_00);

    let mut block_01 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_01.append_row(DynamicVector::from([5.0]));
    block_01.append_row(DynamicVector::from([6.0]));
    block_matrix.set_block(0, 1, block_01);

    let mut block_11 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_11.append_row(DynamicVector::from([7.0]));
    block_matrix.set_block(1, 1, block_11);

    let flat = block_matrix.flatten();
    assert_eq!(flat.num_rows(), 3);
    assert_eq!(flat.num_cols(), 3);

    assert_eq!(*flat.get_component(0, 0), 1.0);
    assert_eq!(*flat.get_component(0, 1), 2.0);
    assert_eq!(*flat.get_component(0, 2), 5.0);
    assert_eq!(*flat.get_component(1, 0), 3.0);
    assert_eq!(*flat.get_component(1, 1), 4.0);
    assert_eq!(*flat.get_component(1, 2), 6.0);
    assert_eq!(*flat.get_component(2, 0), 0.0);
    assert_eq!(*flat.get_component(2, 1), 0.0);
    assert_eq!(*flat.get_component(2, 2), 7.0);
  }

  #[test]
  fn test_iter_blocks() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![1, 1], vec![1, 1]);

    let mut block_00 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_00.append_row(DynamicVector::from([1.0]));
    let mut block_11 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_11.append_row(DynamicVector::from([2.0]));

    block_matrix.set_block(0, 0, block_00);
    block_matrix.set_block(1, 1, block_11);

    let blocks: Vec<_> = block_matrix.iter_blocks().collect();
    assert_eq!(blocks.len(), 2);

    let positions: Vec<_> = blocks.iter().map(|(pos, _)| *pos).collect();
    assert!(positions.contains(&(0, 0)));
    assert!(positions.contains(&(1, 1)));
  }

  #[test]
  fn test_malformed_prevention() {
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2, 3], vec![2, 1]);

    let mut valid_block_00 = DynamicDenseMatrix::<f64, RowMajor>::new();
    valid_block_00.append_row(DynamicVector::from([1.0, 2.0]));
    valid_block_00.append_row(DynamicVector::from([3.0, 4.0]));

    let mut valid_block_10 = DynamicDenseMatrix::<f64, RowMajor>::new();
    valid_block_10.append_row(DynamicVector::from([5.0, 6.0]));
    valid_block_10.append_row(DynamicVector::from([7.0, 8.0]));
    valid_block_10.append_row(DynamicVector::from([9.0, 10.0]));

    block_matrix.set_block(0, 0, valid_block_00);
    block_matrix.set_block(1, 0, valid_block_10);

    let flat = block_matrix.flatten();
    assert_eq!(flat.num_rows(), 5);
    assert_eq!(flat.num_cols(), 3);
    assert_eq!(*flat.get_component(0, 0), 1.0);
    assert_eq!(*flat.get_component(2, 0), 5.0);
    assert_eq!(*flat.get_component(4, 1), 10.0);
  }

  #[test]
  fn test_display_formatting() {
    // Test empty block matrix (all zero blocks)
    let empty_block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2, 1], vec![1, 2]);
    println!("Empty block matrix (all zeros):");
    println!("{empty_block_matrix}");

    // Test block matrix with some non-zero blocks
    let mut block_matrix = BlockMatrix::<f64, RowMajor>::new(vec![2, 2], vec![2, 1]);

    // Create block (0,0)
    let mut block_00 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_00.append_row(DynamicVector::from([1.0, 2.0]));
    block_00.append_row(DynamicVector::from([3.0, 4.0]));
    block_matrix.set_block(0, 0, block_00);

    // Create block (1,1)
    let mut block_11 = DynamicDenseMatrix::<f64, RowMajor>::new();
    block_11.append_row(DynamicVector::from([5.0]));
    block_11.append_row(DynamicVector::from([6.0]));
    block_matrix.set_block(1, 1, block_11);

    println!("Block matrix with some non-zero blocks:");
    println!("{block_matrix}");

    // Test single block matrix
    let mut single_block = BlockMatrix::<f64, RowMajor>::new(vec![3], vec![2]);
    let mut block = DynamicDenseMatrix::<f64, RowMajor>::new();
    block.append_row(DynamicVector::from([10.0, 20.0]));
    block.append_row(DynamicVector::from([30.0, 40.0]));
    block.append_row(DynamicVector::from([50.0, 60.0]));
    single_block.set_block(0, 0, block);

    println!("Single block matrix:");
    println!("{single_block}");
  }
}
