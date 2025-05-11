use std::collections::HashMap;

use crate::{definitions::TopologicalSpace, lattice::Lattice, set::Set, sheaf::Presheaf};

// TODO: This has not been optimized at all, and for certain operations this may be an inefficient
// data structure, but it works for now.

/// A cell in a cell complex, representing a k-dimensional cell with its attachments
/// to (k-1)-dimensional cells.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cell {
  /// The dimension of this cell
  dimension:   usize,
  /// The unique identifier for this cell
  id:          usize,
  /// The cells this cell is attached to (its star)
  attachments: Vec<usize>,
}

impl Cell {
  /// Creates a new cell with the given dimension and attachments.
  ///
  /// # Arguments
  /// * `dimension` - The dimension of the cell
  /// * `id` - A unique identifier for the cell
  /// * `attachments` - The IDs of the (k-1)-cells this cell is attached to
  pub fn new(dimension: usize, id: usize, attachments: Vec<usize>) -> Self {
    Self { dimension, id, attachments }
  }

  /// Returns the dimension of this cell
  pub fn dimension(&self) -> usize { self.dimension }

  /// Returns the unique identifier of this cell
  pub fn id(&self) -> usize { self.id }

  /// Returns a reference to the attachments of this cell
  pub fn attachments(&self) -> &[usize] { &self.attachments }
}

/// A cell complex representing a collection of cells with their attachment relationships.
/// The attachment structure is represented as a lattice where cells are ordered by
/// their attachment relationships.
#[derive(Debug)]
pub struct CellComplex {
  /// The cells in the complex, indexed by their IDs
  cells:              HashMap<usize, Cell>,
  /// The attachment relationships between cells, represented as a lattice
  attachment_lattice: Lattice<usize>,
  /// The next available cell ID
  next_id:            usize,
}

impl CellComplex {
  /// Creates a new empty cell complex
  pub fn new() -> Self {
    Self {
      cells:              HashMap::new(),
      attachment_lattice: Lattice::new(),
      next_id:            0,
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
  pub fn add_cell(&mut self, dimension: usize, attachments: Vec<usize>) -> usize {
    // Validate attachments exist and have correct dimension
    for &att_id in &attachments {
      let att_cell = self.cells.get(&att_id).expect("Attachment cell does not exist");
      assert_eq!(att_cell.dimension(), dimension - 1, "Attachment cell has wrong dimension");
    }

    let id = self.next_id;
    self.next_id += 1;

    let cell = Cell::new(dimension, id, attachments.clone());
    self.cells.insert(id, cell);

    // Add to lattice
    self.attachment_lattice.add_element(id);
    for &att_id in &attachments {
      self.attachment_lattice.add_relation(att_id, id);
    }

    id
  }

  /// Returns a reference to a cell by its ID
  pub fn get_cell(&self, id: usize) -> Option<&Cell> { self.cells.get(&id) }

  /// Returns all cells of a given dimension
  pub fn cells_of_dimension(&self, dimension: usize) -> Vec<&Cell> {
    self.cells.values().filter(|cell| cell.dimension() == dimension).collect()
  }

  /// Returns the boundary of a cell, i.e., all cells it is attached to
  pub fn boundary(&self, id: usize) -> Vec<&Cell> {
    if let Some(cell) = self.cells.get(&id) {
      cell.attachments().iter().filter_map(|&att_id| self.cells.get(&att_id)).collect()
    } else {
      Vec::new()
    }
  }

  /// Returns the open star of a cell, i.e., all cells of dimension (k+1) that are attached to it.
  ///
  /// # Arguments
  /// * `id` - The ID of the cell to find the open star of
  ///
  /// # Returns
  /// A vector of references to cells that are exactly one dimension higher than the given cell
  /// and are attached to it.
  pub fn open_star(&self, id: usize) -> Vec<&Cell> {
    if let Some(cell) = self.cells.get(&id) {
      let target_dim = cell.dimension() + 1;
      self
        .cells
        .values()
        .filter(|other_cell| {
          other_cell.dimension() == target_dim && other_cell.attachments().contains(&id)
        })
        .collect()
    } else {
      Vec::new()
    }
  }

  /// Returns the maximal dimension of any cell in the complex
  pub fn max_dimension(&self) -> usize {
    self.cells.values().map(|cell| cell.dimension()).max().unwrap_or(0)
  }
}

impl Set for CellComplex {
  type Point = todo!();
}

impl TopologicalSpace for CellComplex {
  type OpenSet = todo!();
  type Point = todo!();
}

impl Presheaf<CellComplex> for CellComplex {
  type Data = todo!();
  type Section = todo!();

  fn restrict(
    &self,
    section: &Self::Section,
    from: &<CellComplex as TopologicalSpace>::OpenSet,
    to: &<CellComplex as TopologicalSpace>::OpenSet,
  ) -> Self::Section {
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

    // Check boundary
    assert_eq!(complex.boundary(e1).len(), 1);
    assert_eq!(complex.boundary(t1).len(), 1);

    // Check open star
    assert_eq!(complex.open_star(v1).len(), 1); // v1 has one 1-cell attached
    assert_eq!(complex.open_star(e1).len(), 1); // e1 has one 2-cell attached
    assert_eq!(complex.open_star(t1).len(), 0); // t1 has no 3-cells attached
  }

  #[test]
  #[should_panic(expected = "Attachment cell has wrong dimension")]
  fn test_invalid_attachment_dimension() {
    let mut complex = CellComplex::new();
    let v1 = complex.add_cell(0, vec![]);
    let v2 = complex.add_cell(0, vec![]);

    // Try to attach a 1-cell to two 0-cells (should be valid)
    let e1 = complex.add_cell(1, vec![v1, v2]);

    // Try to attach a 2-cell to a 0-cell (should panic)
    complex.add_cell(2, vec![v1]);
  }
}
