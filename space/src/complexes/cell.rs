//! # Cell Complex Module
//!
//! This module provides data structures and traits for representing regular cell complexes and
//! cellular sheaves, as used in algebraic topology and combinatorial topology.
//!
//! ## Features
//! - Defines the [`Cell`] struct for individual cells (with dimension and attachment information).
//! - Defines the [`CellComplex`] struct for collections of cells and their attachment (face)
//!   relations.
//! - Provides efficient APIs for querying boundaries, stars, and k-skeleta of cell complexes.
//! - (TODO) Implements the [`Set`] and [`TopologicalSpace`] traits for cell complexes.
//! - (TODO) Defines the [`CellularSheaf`] struct for associating stalks and restriction maps to
//!   cells, supporting computations in cellular sheaf theory.
//!
//! This module is suitable for applications in topological data analysis, sheaf theory, and related
//! areas.

use std::{
  cell::RefCell,
  collections::HashMap,
  rc::{Rc, Weak},
};

use crate::{definitions::Topology, lattice::Lattice, set::Collection};

// TODO: This has not been optimized at all, and for certain operations this may be an inefficient
// data structure, but it works for now.

/// A cell in a cell complex, representing a k-dimensional cell with its attachments
/// to (k-1)-dimensional cells.
#[derive(Debug, Clone)]
pub struct Cell {
  /// The dimension of this cell
  dimension: usize,
  /// The unique identifier for this cell
  id:        usize,
  /// The topological space this cell is part of
  space:     Weak<RefCell<CellComplexInner>>,
}

impl Cell {
  /// Returns the dimension of this cell
  pub fn dimension(&self) -> usize { self.dimension }

  /// Returns the unique identifier of this cell
  pub fn id(&self) -> usize { self.id }

  /// Returns the set of cells of dimension `k+1` that this cell is attached to
  pub fn attachments(&self) -> Vec<usize> {
    let space = self.space.upgrade().unwrap();
    let inner = space.borrow();
    inner.attachment_lattice.successors(self.id).into_iter().collect()
  }
}

/// A cell complex representing a collection of cells with their attachment relationships.
/// The attachment structure is represented as a lattice where cells are ordered by
/// their attachment relationships.
#[derive(Debug, Default)]
pub struct CellComplex {
  inner: Rc<RefCell<CellComplexInner>>,
}

#[derive(Debug, Default)]
struct CellComplexInner {
  /// The attachment relationships between cells, represented as a lattice
  attachment_lattice: Lattice<usize>,
  /// The next available cell ID
  next_id:            usize,
  /// The cells in the complex, indexed by their IDs
  cells:              HashMap<usize, Cell>,
}

impl CellComplex {
  /// Creates a new empty cell complex
  pub fn new() -> Self { Self { inner: Rc::new(RefCell::new(CellComplexInner::default())) } }

  /// Adds a new cell to the complex with the given attachments.
  ///
  /// # Arguments
  /// * `dimension` - The dimension of the cell to add
  /// * `attachments` - The IDs of the (k-1)-cells this cell should be attached to
  ///
  /// # Returns
  /// The ID of the newly added cell
  ///
  /// # Panics
  /// * If any of the attachment IDs don't exist in the complex
  /// * If any of the attachments are not of dimension (k-1)
  pub fn add_cell(&mut self, dimension: usize, attachments: Vec<usize>) -> usize {
    // Validate attachments exist and have correct dimension
    for &att_id in &attachments {
      let inner = self.inner.borrow();
      let att_cell = inner.cells.get(&att_id).expect("Attachment cell does not exist");
      assert_eq!(att_cell.dimension(), dimension - 1, "Attachment cell has wrong dimension");
    }

    let mut inner = self.inner.borrow_mut();
    let id = inner.next_id;
    inner.next_id += 1;

    let cell = Cell { space: Rc::downgrade(&self.inner), dimension, id };
    inner.cells.insert(id, cell);

    // Add to lattice
    inner.attachment_lattice.add_element(id);
    for &att_id in &attachments {
      inner.attachment_lattice.add_relation(att_id, id);
    }
    id
  }

  /// Returns a reference to a cell by its ID
  pub fn get_cell(&self, id: usize) -> Option<Cell> {
    let inner = self.inner.borrow();
    inner.cells.get(&id).cloned()
  }

  /// Returns all cells of a given dimension
  pub fn cells_of_dimension(&self, dimension: usize) -> Vec<Cell> {
    self
      .inner
      .borrow()
      .cells
      .values()
      .filter(|cell| cell.dimension() == dimension)
      .cloned()
      .collect()
  }

  /// Returns the maximal dimension of any cell in the complex
  pub fn max_dimension(&self) -> usize {
    self.inner.borrow().cells.values().map(Cell::dimension).max().unwrap_or(0)
  }
}

impl Collection for CellComplex {
  type Item = Cell;

  fn contains(&self, point: &Self::Item) -> bool {
    self.inner.borrow().cells.contains_key(&point.id())
  }

  fn is_empty(&self) -> bool { self.inner.borrow().cells.is_empty() }
}

impl Topology for Cell {
  type Space = CellComplex;

  fn neighborhood(&self) -> Vec<Self> { todo!() }

  fn boundary<R: harness_algebra::prelude::Ring + Copy>(&self) -> crate::homology::Chain<Self, R> {
    todo!()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_basic_cell_complex() {
    let mut complex = CellComplex::new();

    // Add a 0-cell (vertex)
    let v1 = complex.add_cell(0, vec![]);

    // Add a 1-cell (edge) attached to the vertex
    let e1 = complex.add_cell(1, vec![v1]);

    // Add a 2-cell (triangle) attached to the edge
    let t1 = complex.add_cell(2, vec![e1]);

    // Check dimensions
    assert_eq!(complex.get_cell(v1).unwrap().dimension(), 0);
    assert_eq!(complex.get_cell(e1).unwrap().dimension(), 1);
    assert_eq!(complex.get_cell(t1).unwrap().dimension(), 2);

    // Check attachments
    assert_eq!(complex.get_cell(e1).unwrap().attachments(), &[v1]);
    assert_eq!(complex.get_cell(t1).unwrap().attachments(), &[e1]);

    todo!()
    // // Check boundary
    // assert_eq!(complex.boundary(e1).len(), 1);
    // assert_eq!(complex.boundary(t1).len(), 1);

    // // Check open star
    // assert_eq!(complex.star(v1).len(), 1); // v1 has one 1-cell attached
    // assert_eq!(complex.star(e1).len(), 1); // e1 has one 2-cell attached
    // assert_eq!(complex.star(t1).len(), 0); // t1 has no 3-cells attached
  }

  #[test]
  #[should_panic(expected = "Attachment cell has wrong dimension")]
  fn test_invalid_attachment_dimension() {
    let mut complex = CellComplex::new();
    let v1 = complex.add_cell(0, vec![]);
    let v2 = complex.add_cell(0, vec![]);

    // Try to attach a 1-cell to two 0-cells (should be valid)
    let _e1 = complex.add_cell(1, vec![v1, v2]);

    // Try to attach a 2-cell to a 0-cell (should panic)
    complex.add_cell(2, vec![v1]);
  }
}
