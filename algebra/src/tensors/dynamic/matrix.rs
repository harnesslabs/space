use std::{fmt::Debug, marker::PhantomData};

use super::{vector::DynamicVector, *};

// Struct to store pivot coordinates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PivotInfo {
  pub row: usize,
  pub col: usize,
}

// Struct to store the result of row echelon form computation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEchelonOutput {
  pub rank:   usize,
  pub pivots: Vec<PivotInfo>,
}

// Sealed trait pattern to prevent external implementations of MatrixOrientation
mod sealed {
  pub trait Sealed {}
  impl Sealed for super::RowMajor {}
  impl Sealed for super::ColumnMajor {}
}

/// A marker trait for matrix storage orientation (RowMajor or ColumnMajor).
/// This trait is sealed, meaning only types defined in this crate can implement it.
pub trait MatrixOrientation: sealed::Sealed {}

/// Marker type for row-major matrix storage.
/// In row-major storage, elements of a row are contiguous in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowMajor;
impl MatrixOrientation for RowMajor {}

/// Marker type for column-major matrix storage.
/// In column-major storage, elements of a column are contiguous in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnMajor;
impl MatrixOrientation for ColumnMajor {}

/// A dynamically-sized matrix (typically with components from a field `F`).
///
/// The dimensions can be determined at runtime, making it flexible for various applications.
/// The matrix can be either RowMajor or ColumnMajor, specified by the `O` type parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicDenseMatrix<F, O: MatrixOrientation = RowMajor> {
  /// For RowMajor: components is a DynamicVector of rows (each row is a DynamicVector<F>).
  /// For ColumnMajor: components is a DynamicVector of columns (each col is a DynamicVector<F>).
  components:  DynamicVector<DynamicVector<F>>,
  orientation: PhantomData<O>,
}

impl<F, O: MatrixOrientation> DynamicDenseMatrix<F, O> {
  /// Creates a new, empty DynamicMatrix with the specified orientation.
  pub fn new() -> Self {
    Self { components: DynamicVector::new(Vec::new()), orientation: PhantomData }
  }
}

impl<F: Field + Copy> DynamicDenseMatrix<F, RowMajor> {
  /// Returns the number of rows in the matrix.
  pub fn num_rows(&self) -> usize {
    self.components.dimension() // Outer vector stores rows
  }

  /// Returns the number of columns in the matrix.
  /// Assumes a non-ragged matrix if rows > 0.
  pub fn num_cols(&self) -> usize {
    if self.components.dimension() == 0 {
      0
    } else {
      self.components.components()[0].dimension()
    }
  }

  /// Appends a new column to the matrix.
  /// The column's length must match the number of existing rows.
  pub fn append_column(&mut self, column: DynamicVector<F>) {
    let num_r = self.num_rows();
    if num_r == 0 {
      // Matrix is empty, this column defines the rows and the first column
      if column.dimension() == 0 {
        return; // Appending an empty column to an empty matrix results in an empty matrix
      }
      // Each element of the input column becomes a new row with one element.
      for i in 0..column.dimension() {
        self.components.components_mut().push(DynamicVector::new(vec![*column.get_component(i)]));
      }
    } else {
      assert_eq!(num_r, column.dimension(), "Column length must match the number of rows");
      for i in 0..num_r {
        // self.components.components is Vec<DynamicVector<F>> (rows)
        // Each self.components.components[i] is a DynamicVector<F> (a row)
        // We need to append to its internal Vec<F>.
        self.components.components_mut()[i].append(*column.get_component(i));
      }
    }
  }

  /// Returns a new DynamicVector representing the column at the given index.
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
  /// The column's length must match the number of existing rows.
  pub fn set_column(&mut self, index: usize, column: DynamicVector<F>) {
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
  /// If the matrix is not empty, the new row's length must match the number of existing columns.
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
  pub fn get_row(&self, index: usize) -> &DynamicVector<F> {
    assert!(index < self.num_rows(), "Row index out of bounds");
    &self.components.components()[index]
  }

  /// Sets the row at the given index with the provided DynamicVector.
  pub fn set_row(&mut self, index: usize, row: DynamicVector<F>) {
    assert!(index < self.num_rows(), "Row index out of bounds");
    if self.num_rows() > 0 {
      // Or check if self.components.dimension() > 0 before num_cols()
      assert_eq!(
        self.num_cols(),
        row.dimension(),
        "New row length must match existing number of columns"
      );
    }
    self.components.components_mut()[index] = row;
  }

  /// Returns the component at the given row and column.
  pub fn get_component(&self, row: usize, col: usize) -> &F {
    assert!(row < self.num_rows(), "Row index out of bounds");
    assert!(col < self.num_cols(), "Column index out of bounds"); // Relies on num_cols correctly assessing based on first row
    self.components.components()[row].get_component(col)
  }

  /// Sets the component at the given row and column to the given value.
  pub fn set_component(&mut self, row: usize, col: usize, value: F) {
    assert!(row < self.num_rows(), "Row index out of bounds");
    assert!(col < self.num_cols(), "Column index out of bounds");
    self.components.components_mut()[row].set_component(col, value);
  }

  pub fn transpose(self) -> DynamicDenseMatrix<F, ColumnMajor> {
    DynamicDenseMatrix { components: self.components, orientation: PhantomData }
  }

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
      // For a field, a non-zero pivot_val is expected here.
      // The Field trait should provide inverse. Panicking if not found for a non-zero element is
      // acceptable.
      let inv_pivot = pivot_val.multiplicative_inverse();

      // Normalize pivot row: matrix[r][j] = matrix[r][j] * inv_pivot
      for j in lead..cols {
        let val = *matrix[r].get_component(j);
        matrix[r].set_component(j, val * inv_pivot);
      }

      // Eliminate other rows: matrix[i] = matrix[i] - factor * matrix[r]
      for i_row in 0..rows {
        if i_row != r {
          let factor = *matrix[i_row].get_component(lead); // factor is a copy
          if !factor.is_zero() {
            for j_col in lead..cols {
              let val_r_j_col = *matrix[r].get_component(j_col); // Read from pivot row (r)
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
}

impl<F: Field + Copy> DynamicDenseMatrix<F, ColumnMajor> {
  /// Returns the number of rows in the matrix.
  /// For ColumnMajor, this is the dimension of the first column vector (if any).
  pub fn num_rows(&self) -> usize {
    if self.components.dimension() == 0 {
      // No columns
      0
    } else {
      // All columns should have the same number of rows (length)
      self.components.components()[0].dimension()
    }
  }

  /// Returns the number of columns in the matrix.
  /// For ColumnMajor, this is the number of column vectors stored.
  pub fn num_cols(&self) -> usize {
    self.components.dimension() // Outer vector stores columns
  }

  /// Appends a new column to the matrix.
  /// If the matrix is not empty, the new column's length must match the number of existing rows.
  pub fn append_column(&mut self, column: DynamicVector<F>) {
    if self.num_cols() > 0 {
      // Matrix is not empty
      assert_eq!(
        self.num_rows(),
        column.dimension(),
        "New column length must match existing number of rows"
      );
    }
    self.components.components_mut().push(column); // Add the new column vector
  }

  /// Returns a reference to the column at the given index.
  pub fn get_column(&self, index: usize) -> &DynamicVector<F> {
    assert!(index < self.num_cols(), "Column index out of bounds");
    &self.components.components()[index]
  }

  /// Sets the column at the given index with the provided DynamicVector.
  pub fn set_column(&mut self, index: usize, column: DynamicVector<F>) {
    assert!(index < self.num_cols(), "Column index out of bounds");
    if self.num_cols() > 0 {
      // Or check self.components.dimension() > 0
      assert_eq!(
        self.num_rows(),
        column.dimension(),
        "New column length must match existing number of rows"
      );
    }
    self.components.components_mut()[index] = column;
  }

  /// Appends a new row to the matrix.
  /// The row's length must match the number of existing columns.
  pub fn append_row(&mut self, row: DynamicVector<F>) {
    let num_c = self.num_cols();
    if num_c == 0 {
      // Matrix is empty, this row defines the columns and the first row
      if row.dimension() == 0 {
        return; // Appending an empty row to an empty matrix results in an empty matrix
      }
      // Each element of the input row becomes a new column with one element.
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
  /// The row's length must match the number of existing columns.
  pub fn set_row(&mut self, index: usize, row: DynamicVector<F>) {
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
  pub fn get_component(&self, row: usize, col: usize) -> &F {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds");
    self.components.components()[col].get_component(row)
  }

  /// Sets the component at the given row and column to the given value.
  pub fn set_component(&mut self, row: usize, col: usize, value: F) {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds");
    self.components.components_mut()[col].set_component(row, value);
  }

  pub fn transpose(self) -> DynamicDenseMatrix<F, RowMajor> {
    DynamicDenseMatrix { components: self.components, orientation: PhantomData }
  }

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
          for k_col in 0..num_actual_cols {
            // Iterate through all columns to swap elements
            let temp = *matrix[k_col].get_component(i_search_row);
            let val_at_pivot_row = *matrix[k_col].get_component(pivot_row_idx);
            matrix[k_col].set_component(i_search_row, val_at_pivot_row);
            matrix[k_col].set_component(pivot_row_idx, temp);
          }
        }

        pivots.push(PivotInfo { row: pivot_row_idx, col: pivot_col_idx });

        let pivot_val = *matrix[pivot_col_idx].get_component(pivot_row_idx);

        if !pivot_val.is_zero() {
          let inv_pivot_val = pivot_val.multiplicative_inverse();
          for k_col in pivot_col_idx..num_actual_cols {
            let current_val = *matrix[k_col].get_component(pivot_row_idx);
            matrix[k_col].set_component(pivot_row_idx, current_val * inv_pivot_val);
          }
        }

        for k_row in 0..num_actual_rows {
          if k_row != pivot_row_idx {
            let factor = *matrix[pivot_col_idx].get_component(k_row);
            if !factor.is_zero() {
              for j_col_elim in pivot_col_idx..num_actual_cols {
                let val_from_pivot_row_at_j_col = *matrix[j_col_elim].get_component(pivot_row_idx);
                let term_to_subtract = factor * val_from_pivot_row_at_j_col;
                let current_val_in_k_row_at_j_col = *matrix[j_col_elim].get_component(k_row);
                matrix[j_col_elim]
                  .set_component(k_row, current_val_in_k_row_at_j_col - term_to_subtract);
              }
            }
          }
        }
        pivot_row_idx += 1;
        rank += 1;
      }
    }
    RowEchelonOutput { rank, pivots }
  }
}

#[cfg(test)]
mod tests {
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
    m.append_column(c0.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 1);
    assert_eq!(m.get_column(0).components(), &c0_data);
    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(1, 0), 2.0);

    // Append second column
    let c1_data = vec![3.0, 4.0];
    let c1 = DynamicVector::new(c1_data.clone());
    m.append_column(c1.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_column(1).components(), &c1_data);
    assert_eq!(*m.get_component(0, 1), 3.0);
    assert_eq!(*m.get_component(1, 1), 4.0);

    // Set a column
    let c_new_data = vec![5.0, 6.0];
    let c_new = DynamicVector::new(c_new_data.clone());
    m.set_column(0, c_new.clone());
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
    m.append_row(r0.clone());
    assert_eq!(m.num_rows(), 1);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(0).components(), &r0_data);
    assert_eq!(*m.get_component(0, 0), 1.0);
    assert_eq!(*m.get_component(0, 1), 2.0);

    // Append second row
    let r1_data = vec![3.0, 4.0];
    let r1 = DynamicVector::new(r1_data.clone());
    m.append_row(r1.clone());
    assert_eq!(m.num_rows(), 2);
    assert_eq!(m.num_cols(), 2);
    assert_eq!(m.get_row(1).components(), &r1_data);
    assert_eq!(*m.get_component(1, 0), 3.0);
    assert_eq!(*m.get_component(1, 1), 4.0);

    // Set a row
    let r_new_data = vec![5.0, 6.0];
    let r_new = DynamicVector::new(r_new_data.clone());
    m.set_row(0, r_new.clone());
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
    m.append_column(DynamicVector::new(vec![1.0, 2.0]));
    m.append_column(DynamicVector::new(vec![3.0])); // Should panic
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
    m.append_column(DynamicVector::new(vec![1.0]));
    m.set_column(0, DynamicVector::new(vec![3.0, 4.0, 5.0])); // Should panic
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
    m.append_row(DynamicVector::new(vec![1.0, 2.0]));
    m.append_row(DynamicVector::new(vec![3.0])); // Should panic
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
}
