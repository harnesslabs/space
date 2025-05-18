use std::{fmt::Debug, marker::PhantomData};

use super::{vector::DynamicVector, *};

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
      // Get the first row and return its dimension (length)
      // Requires access to DynamicVector::components or a method giving an inner vector by ref.
      // Assuming self.components.components gives access to Vec<DynamicVector<F>> for RowMajor.
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
        self
          .components
          .components_mut()
          .to_vec()
          .push(DynamicVector::new(vec![column.get_component(i)]));
      }
    } else {
      assert_eq!(num_r, column.dimension(), "Column length must match the number of rows");
      for i in 0..num_r {
        // self.components.components is Vec<DynamicVector<F>> (rows)
        // Each self.components.components[i] is a DynamicVector<F> (a row)
        // We need to append to its internal Vec<F>.
        self.components.components_mut()[i].append(column.get_component(i));
      }
    }
  }

  /// Returns a new DynamicVector representing the column at the given index.
  pub fn get_column(&self, index: usize) -> DynamicVector<F> {
    let num_r = self.num_rows();
    if num_r == 0 {
      return DynamicVector::new(vec![]);
    }
    assert!(index < self.num_cols(), "Column index out of bounds");
    let mut col_components = Vec::with_capacity(num_r);
    for i in 0..num_r {
      col_components.push(self.components.components()[i].get_component(index));
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
      self.components.components_mut()[i].set_component(index, column.get_component(i));
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
    self.components.components_mut().to_vec().push(row);
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
  pub fn get_component(&self, row: usize, col: usize) -> F {
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
    self.components.components_mut().to_vec().push(column); // Add the new column vector
  }

  /// Returns a new DynamicVector representing the column at the given index.
  pub fn get_column(&self, index: usize) -> DynamicVector<F> {
    assert!(index < self.num_cols(), "Column index out of bounds");
    // For ColumnMajor, a column is directly one of the inner DynamicVectors.
    // We need to return a new owned DynamicVector, so clone its components.
    DynamicVector::new(self.components.components()[index].components().to_vec())
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
        self
          .components
          .components_mut()
          .to_vec()
          .push(DynamicVector::new(vec![row.get_component(i)]));
      }
    } else {
      assert_eq!(num_c, row.dimension(), "Row length must match the number of columns");
      for i in 0..num_c {
        // self.components.components[i] is a column (DynamicVector<F>)
        // Append the i-th element of the new row to the i-th column vector.
        self.components.components_mut()[i].components().to_vec().push(row.get_component(i));
      }
    }
  }

  /// Returns a new DynamicVector representing the row at the given index.
  pub fn get_row(&self, index: usize) -> DynamicVector<F> {
    let num_c = self.num_cols();
    if num_c == 0 {
      return DynamicVector::new(vec![]); // No columns means no rows effectively
    }
    assert!(index < self.num_rows(), "Row index out of bounds");
    let mut row_components = Vec::with_capacity(num_c);
    for i in 0..num_c {
      row_components.push(self.components.components()[i].get_component(index));
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
      self.components.components_mut()[i].set_component(index, row.get_component(i));
    }
  }

  /// Returns the component at the given row and column.
  pub fn get_component(&self, row: usize, col: usize) -> F {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds"); // Relies on num_rows correctly assessing based on first col
    self.components.components()[col].get_component(row) // Access col first, then row
  }

  /// Sets the component at the given row and column to the given value.
  pub fn set_component(&mut self, row: usize, col: usize, value: F) {
    assert!(col < self.num_cols(), "Column index out of bounds");
    assert!(row < self.num_rows(), "Row index out of bounds");
    self.components.components_mut()[col].set_component(row, value);
  }
}
