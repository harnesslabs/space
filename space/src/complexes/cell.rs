//! # Cell Complex Module
//!
//! This module provides data structures and traits for representing regular cell complexes and
//! cellular sheaves, as used in algebraic topology and combinatorial topology.
//!
//! ## Features
//! - Defines the [`Cell`] struct for individual cells (with dimension and attachment information).
//! - Defines the [`CellComplex`] struct for collections of cells and their attachment (face)
//!   relations.
//!
//! This module is suitable for applications in topological data analysis, sheaf theory, and related
//! areas.

use std::collections::HashMap;

use super::*;
use crate::{
  definitions::Topology,
  lattice::Lattice,
  set::{Collection, Poset},
};

// TODO: This has not been optimized at all, and for certain operations this may be an inefficient
// data structure, but it works for now. We can likely store multiple maps to reference cells more
// quickly.
// TODO: It's worth noting that by in large this is just a lattice with some extra data and
// requirements.

/// A cell in a cell complex, representing a k-dimensional cell with its attachments
/// to (k-1)-dimensional cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
  /// The dimension of this cell
  pub dimension: usize,
  /// The unique identifier for this cell
  pub id:        usize,
}

impl Cell {
  /// Returns the dimension of this cell
  pub const fn dimension(&self) -> usize { self.dimension }

  /// Returns the unique identifier of this cell
  pub const fn id(&self) -> usize { self.id }
}

/// A cell complex representing a collection of cells with their attachment relationships.
/// The attachment structure is represented as a lattice where cells are ordered by
/// their attachment relationships.
#[derive(Debug, Default, Clone)]
pub struct CellComplex {
  /// The attachment relationships between cells, represented as a lattice
  pub attachment_lattice: Lattice<usize>,
  /// The next available cell ID
  pub next_id:            usize,
  /// The cells in the complex, indexed by their IDs
  pub cells:              HashMap<usize, Cell>,
}

impl CellComplex {
  /// Creates a new empty cell complex
  pub fn new() -> Self {
    Self {
      attachment_lattice: Lattice::new(),
      next_id:            0,
      cells:              HashMap::new(),
    }
  }

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
  pub fn add_cell_by_id(&mut self, dimension: usize, attachments: &[usize]) -> Cell {
    // Validate attachments exist and have correct dimension
    for &att_id in attachments {
      let att_cell = self.cells.get(&att_id).expect("Attachment cell does not exist");
      assert_eq!(att_cell.dimension(), dimension - 1, "Attachment cell has wrong dimension");
    }

    let id = self.next_id;
    self.next_id += 1;

    let cell = Cell { dimension, id };
    self.cells.insert(id, cell.clone());

    // Add to lattice
    self.attachment_lattice.add_element(id);
    for &att_id in attachments {
      self.attachment_lattice.add_relation(att_id, id);
    }
    cell
  }

  pub fn add_cell(&mut self, dimension: usize, attachments: &[&Cell]) -> Cell {
    let ids = attachments.iter().map(|cell| cell.id()).collect::<Vec<_>>();
    self.add_cell_by_id(dimension, &ids)
  }

  pub fn get_cell(&self, id: usize) -> Option<Cell> { self.cells.get(&id).cloned() }

  pub fn cells_of_dimension(&self, dimension: usize) -> Vec<Cell> {
    self.cells.values().filter(|cell| cell.dimension() == dimension).cloned().collect()
  }

  pub fn max_dimension(&self) -> usize {
    self.cells.values().map(Cell::dimension).max().unwrap_or(0)
  }

  pub fn get_attachments(&self, cell: &Cell) -> Vec<Cell> {
    self
      .attachment_lattice
      .successors(cell.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  pub fn faces(&self, cell: &Cell) -> Vec<Cell> {
    self
      .attachment_lattice
      .predecessors(cell.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }
}

impl Collection for CellComplex {
  type Item = Cell;

  fn contains(&self, point: &Self::Item) -> bool { self.cells.contains_key(&point.id()) }

  fn is_empty(&self) -> bool { self.cells.is_empty() }
}

impl Poset for CellComplex {
  fn leq(&self, a: &Self::Item, b: &Self::Item) -> Option<bool> {
    self.attachment_lattice.leq(&a.id, &b.id)
  }

  fn upset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self.attachment_lattice.upset(a.id()).into_iter().map(|id| self.get_cell(id).unwrap()).collect()
  }

  fn downset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .downset(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  fn minimal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .minimal_elements()
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  fn maximal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .maximal_elements()
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  fn join(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    self.attachment_lattice.join(a.id(), b.id()).map(|id| self.get_cell(id).unwrap())
  }

  fn meet(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    self.attachment_lattice.meet(a.id(), b.id()).map(|id| self.get_cell(id).unwrap())
  }

  fn successors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .successors(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  fn predecessors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .predecessors(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }
}

impl Topology for CellComplex {
  fn neighborhood(&self, item: &Self::Item) -> Vec<Self::Item> {
    self
      .attachment_lattice
      .successors(item.id())
      .into_iter()
      .map(|id| self.get_cell(id).unwrap())
      .collect()
  }

  fn boundary<R: Ring>(&self, item: &Self::Item) -> crate::homology::Chain<Self, R> {
    let mut boundary_chain_items = Vec::with_capacity(item.dimension() + 1);
    let mut boundary_chain_coeffs = Vec::with_capacity(item.dimension() + 1);
    let cell = self.get_cell(item.id()).unwrap();
    let faces = self.faces(&cell);
    for face in faces {
      let coeff = if face.dimension() == item.dimension() - 1 { R::one() } else { -R::one() };
      boundary_chain_items.push(face);
      boundary_chain_coeffs.push(coeff);
    }
    crate::homology::Chain::from_items_and_coeffs(self, boundary_chain_items, boundary_chain_coeffs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_basic_cell_complex() {
    let mut complex = CellComplex::new();

    // Add a 0-cell (vertex)
    let v1 = complex.add_cell(0, &[]);

    // Add a 1-cell (edge) attached to the vertex
    let e1 = complex.add_cell(1, &[&v1]);

    // Add a 2-cell (triangle) attached to the edge
    let t1 = complex.add_cell(2, &[&e1]);

    // Check dimensions5
    assert_eq!(v1.dimension(), 0);
    assert_eq!(e1.dimension(), 1);
    assert_eq!(t1.dimension(), 2);
    assert_eq!(complex.get_cell(v1.id()).unwrap().id(), v1.id());
    assert_eq!(complex.get_cell(e1.id()).unwrap().id(), e1.id());
    assert_eq!(complex.get_cell(t1.id()).unwrap().id(), t1.id());

    // Check attachments
    assert_eq!(complex.get_attachments(&e1), &[t1.clone()]);
    assert_eq!(complex.get_attachments(&t1), &[]);

    // Check neighborhood (open star)
    assert_eq!(complex.neighborhood(&v1).len(), 1); // v1 has one 1-cell attached
    assert_eq!(complex.neighborhood(&e1).len(), 1); // e1 has one 2-cell attached
    assert_eq!(complex.neighborhood(&t1).len(), 0); // t1 has no 3-cells attached
  }

  #[test]
  #[should_panic(expected = "Attachment cell has wrong dimension")]
  fn test_invalid_attachment_dimension() {
    let mut complex = CellComplex::new();
    let v1 = complex.add_cell(0, &[]);
    let v2 = complex.add_cell(0, &[]);

    // Try to attach a 1-cell to two 0-cells (should be valid)
    let _e1 = complex.add_cell(1, &[&v1, &v2]);

    // Try to attach a 2-cell to a 0-cell (should panic)
    complex.add_cell(2, &[&v1]);
  }

  #[test]
  fn test_cell_complex_faces() {
    let mut complex = CellComplex::new();

    // Add a 0-cell (vertex)
    let v1 = complex.add_cell(0, &[]);
    let v2 = complex.add_cell(0, &[]);

    // Add a 1-cell (edge) attached to the vertices
    let e1 = complex.add_cell(1, &[&v1, &v2]);

    // Check faces of the edge
    let faces = complex.faces(&e1);
    assert_eq!(faces.len(), 2);
    assert!(faces.contains(&v1));
    assert!(faces.contains(&v2));
  }
}
