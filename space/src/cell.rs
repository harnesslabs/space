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

use std::collections::{HashMap, HashSet};

use harness_algebra::{
  rings::Field,
  tensors::dynamic::{
    matrix::{DynamicDenseMatrix, RowMajor},
    vector::DynamicVector,
  },
};

use crate::{
  definitions::TopologicalSpace,
  lattice::Lattice,
  set::{Collection, Set},
  sheaf::{Presheaf, Section},
};

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
#[derive(Debug, Default)]
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
    self.cells.get(&id).map_or_else(Vec::new, |cell| {
      let target_dim = cell.dimension() + 1;
      self
        .cells
        .values()
        .filter(|other_cell| {
          other_cell.dimension() == target_dim && other_cell.attachments().contains(&id)
        })
        .collect()
    })
  }

  /// Returns the maximal dimension of any cell in the complex
  pub fn max_dimension(&self) -> usize {
    self.cells.values().map(Cell::dimension).max().unwrap_or(0)
  }
}

impl Collection for CellComplex {
  type Point = Cell;

  fn contains(&self, point: &Self::Point) -> bool { self.cells.contains_key(&point.id()) }

  fn is_empty(&self) -> bool { self.cells.is_empty() }
}

impl Set for CellComplex {
  fn minus(&self, _: &Self) -> Self { todo!() }

  fn meet(&self, _: &Self) -> Self { todo!() }

  fn join(&self, _: &Self) -> Self { todo!() }
}

impl TopologicalSpace for CellComplex {
  type OpenSet = HashSet<Cell>;
  type Point = Cell;

  fn neighborhood(&self, _point: Self::Point) -> Self::OpenSet { todo!() }

  fn is_open(&self, _open_set: Self::OpenSet) -> bool { todo!() }
}

impl<F: Field + Copy, S: ::std::hash::BuildHasher> Section<CellComplex>
  for HashMap<Cell, DynamicVector<F>, S>
{
  type Stalk = DynamicVector<F>;

  fn evaluate(&self, _point: &<CellComplex as TopologicalSpace>::Point) -> Option<Self::Stalk> {
    todo!()
  }

  fn domain(&self) -> HashSet<Cell> { todo!() }

  fn from_closure<G>(_domain: <CellComplex as TopologicalSpace>::OpenSet, _f: G) -> Self
  where G: Fn(&<CellComplex as TopologicalSpace>::Point) -> Option<Self::Stalk> {
    todo!()
  }
}

// TODO (autoparallel): This should be generic over a matrix type eventually, but for now the dense
// row major matrix works.
/// A cellular sheaf over a cell complex, associating data (stalks) and restriction maps to the
/// cells.
///
/// This struct represents a cellular sheaf as used in algebraic topology and sheaf theory, where:
/// - Each cell in the underlying `CellComplex` is assigned a stalk (typically a vector space or
///   module).
/// - Each pair of incident cells (face relation) is assigned a restriction map (matrix) between
///   their stalks.
///
/// # Fields
/// * `complex` - The underlying cell complex over which the sheaf is defined.
/// * `stalk_dimensions` - A mapping from each cell to the dimension of its stalk (e.g., the
///   dimension of the vector space assigned to the cell).
/// * `restriction_matrices` - A mapping from pairs of cell IDs `(from, to)` (where `to` is a face
///   of `from`) to the matrix representing the restriction map from the stalk of `from` to the
///   stalk of `to`.
///
/// This structure is suitable for computations in cellular sheaf theory, such as cohomology, and is
/// compatible with the `Presheaf` trait for functorial operations.
#[derive(Debug)]
pub struct CellularSheaf<F> {
  /// The underlying cell complex over which the sheaf is defined.
  pub complex:              CellComplex,
  /// The dimension of the stalk (vector space) assigned to each cell.
  pub stalk_dimensions:     HashMap<Cell, usize>,
  /// The restriction maps between stalks, represented as matrices.
  /// Each key is a pair of cell IDs `(from, to)` where `to` is a face of `from`,
  /// and the value is the matrix (as a Vec of Vecs) representing the restriction map.
  pub restriction_matrices: HashMap<(usize, usize), DynamicDenseMatrix<F, RowMajor>>,
}

impl<F: Field + Copy> Presheaf<CellComplex> for CellularSheaf<F> {
  type Data = DynamicVector<F>;
  type Section = HashMap<Cell, DynamicVector<F>>;

  fn restrict(
    &self,
    _section: &Self::Section,
    _from: &<CellComplex as TopologicalSpace>::OpenSet,
    _to: &<CellComplex as TopologicalSpace>::OpenSet,
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
    let _e1 = complex.add_cell(1, vec![v1, v2]);

    // Try to attach a 2-cell to a 0-cell (should panic)
    complex.add_cell(2, vec![v1]);
  }
}
