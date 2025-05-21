//! # Cell Complexes and Cellular Topology
//!
//! This module provides data structures for representing **cell complexes**, which are
//! fundamental objects in algebraic topology and combinatorial topology. Cell complexes
//! (or CW complexes) are topological spaces built by attaching $k$-dimensional disks
//! (called $k$-cells) to a $(k-1)$-dimensional skeleton.
//!
//! ## Core Components
//!
//! - **[`Cell`]**: Represents an individual $k$-cell. Each cell has a dimension and a unique
//!   identifier. Examples:
//!     - $0$-cells: points (vertices)
//!     - $1$-cells: line segments (edges)
//!     - $2$-cells: polygons (faces)
//!     - $3$-cells: polyhedra
//!
//! - **[`CellComplex`]**: Represents the entire complex. It is a collection of [`Cell`] objects
//!   along with their attachment information. The attachment structure is crucial and dictates the
//!   topology of the space.
//!   - In this implementation, the `CellComplex` uses an underlying [`Lattice<usize>`]
//!     (`attachment_lattice`) to manage the face/coface (or subcell/supercell) relationships. A
//!     relation `a < b` in the lattice means cell `a` is a face of cell `b`.
//!   - The complex keeps track of cells by their unique IDs and can retrieve cells of a specific
//!     dimension.
//!
//! ## Traits and Functionality
//!
//! The `CellComplex` implements several key traits from the `harness_space` crate:
//! - [`Collection`]: Provides basic set-like operations (e.g., checking if a cell is in the
//!   complex).
//! - [`Poset`]: Leverages the `attachment_lattice` to provide a partial order on cells based on the
//!   face relations. This allows operations like finding upsets (supercells) and downsets
//!   (subcells).
//! - [`Topology`]: Defines topological operations. For `CellComplex`:
//!     - `neighborhood(cell)`: Returns the cells directly attached to `cell` that are of higher
//!       dimension (cofaces or open star, depending on interpretation).
//!     - `boundary(cell)`: Computes the boundary of a cell as a formal sum of its faces (typically
//!       $(k-1)$-cells). The specific coefficients in this implementation depend on its definition
//!       (see trait docs).
//!
//! ## Usage
//!
//! Cell complexes are versatile and can represent a wide variety of topological spaces, often more
//! efficiently than simplicial complexes for certain structures. They are used in:
//! - Computing cellular homology.
//! - Defining cellular sheaves and studying their cohomology.
//! - Applications in topological data analysis, computer graphics, and physics.
//!
//! ## TODOs and Considerations from Original Code
//! - The original code notes that the data structure might not be optimized for all operations and
//!   that it closely resembles a lattice with extra data.
//! - The boundary operator's coefficients in the `Topology` trait implementation for `CellComplex`
//!   might need to be carefully aligned with standard definitions depending on the desired homology
//!   theory (e.g., cellular homology often requires incidence numbers).

use std::collections::HashMap;

use super::*;
use crate::{
  definitions::Topology,
  homology::Chain,
  lattice::Lattice,
  set::{Collection, Poset},
};

// TODO: This has not been optimized at all, and for certain operations this may be an inefficient
// data structure, but it works for now. We can likely store multiple maps to reference cells more
// quickly.
// TODO: It's worth noting that by in large this is just a lattice with some extra data and
// requirements.

/// Represents an individual cell in a cell complex.
///
/// A $k$-cell is a topological space homeomorphic to an open $k$-dimensional disk.
/// In a [`CellComplex`], cells are the basic building blocks, and their attachments define
/// the overall topology of the space.
///
/// Each cell is characterized by its dimension and a unique identifier within the complex.
///
/// # Fields
///
/// * `dimension`: The intrinsic dimension of the cell (e.g., 0 for a vertex, 1 for an edge).
/// * `id`: A unique `usize` identifier for this cell within its parent `CellComplex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
  /// The dimension of this cell (e.g., 0 for a point, 1 for an edge, 2 for a face).
  pub dimension: usize,
  /// A unique identifier for this cell within the context of a [`CellComplex`].
  pub id:        usize,
}

impl Cell {
  /// Returns the dimension of this cell.
  ///
  /// For example, a vertex (0-cell) has dimension 0, an edge (1-cell) has dimension 1.
  pub const fn dimension(&self) -> usize { self.dimension }

  /// Returns the unique identifier of this cell.
  ///
  /// This ID is used to reference the cell within its parent [`CellComplex`]
  /// and its associated `attachment_lattice`.
  pub const fn id(&self) -> usize { self.id }
}

/// Represents a cell complex, a topological space built by attaching cells of various dimensions.
///
/// A cell complex $X$ is constructed inductively. The 0-skeleton $X^0$ is a discrete set of points
/// (0-cells). The $k$-skeleton $X^k$ is formed by attaching $k$-cells to the $(k-1)$-skeleton
/// $X^{k-1}$ via attaching maps.
///
/// This implementation uses an `attachment_lattice` (a [`Lattice<usize>`]) to store the face
/// relationships between cells, where cells are identified by unique `usize` IDs.
/// If cell $a$ is a face of cell $b$, then $a < b$ in the lattice.
///
/// # Fields
///
/// * `attachment_lattice`: A [`Lattice<usize>`] storing the partial order of face relations. The
///   elements in the lattice are cell IDs.
/// * `next_id`: The next available unique ID to assign to a new cell.
/// * `cells`: A `HashMap` mapping cell IDs (`usize`) to [`Cell`] structs, allowing retrieval of
///   cell data (like dimension) by ID.
#[derive(Debug, Default, Clone)]
pub struct CellComplex {
  /// The attachment relationships between cells, represented as a lattice of cell IDs.
  /// If `id_a` is a face of `id_b`, then `attachment_lattice.leq(id_a, id_b)` is true.
  pub attachment_lattice: Lattice<usize>,
  /// The counter for the next available unique cell identifier.
  pub next_id:            usize,
  /// A map storing all cells in the complex, keyed by their unique `usize` ID.
  pub cells:              HashMap<usize, Cell>,
}

impl CellComplex {
  /// Creates a new, empty cell complex.
  ///
  /// Initializes with an empty attachment lattice, no cells, and `next_id` set to 0.
  pub fn new() -> Self {
    Self {
      attachment_lattice: Lattice::new(),
      next_id:            0,
      cells:              HashMap::new(),
    }
  }

  /// Adds a new cell to the complex with specified attachments by their IDs.
  ///
  /// This is a core method for building the cell complex. It creates a new cell of the given
  /// `dimension`, assigns it a unique ID, and establishes face relations in the
  /// `attachment_lattice` between the new cell and its specified `attachments` (which are its
  /// maximal faces).
  ///
  /// # Arguments
  ///
  /// * `dimension`: The dimension of the new cell to add.
  /// * `attachments`: A slice of `usize` IDs representing the cells to which the new cell attaches.
  ///   These are typically expected to be $(d-1)$-cells if the new cell is a $d$-cell.
  ///
  /// # Returns
  ///
  /// The newly created [`Cell`] struct.
  ///
  /// # Panics
  ///
  /// * If any attachment ID in `attachments` does not correspond to an existing cell in the
  ///   complex.
  /// * If any attached cell does not have dimension `dimension - 1` (i.e., not a maximal proper
  ///   face).
  pub fn add_cell_by_id(&mut self, dimension: usize, attachments: &[usize]) -> Cell {
    // Validate attachments exist and have correct dimension
    if dimension > 0 {
      // 0-cells have no attachments in this context
      for &att_id in attachments {
        let att_cell = self.cells.get(&att_id).expect("Attachment cell does not exist");
        assert_eq!(
          att_cell.dimension(),
          dimension - 1,
          "Attachment cell has wrong dimension for face"
        );
      }
    }

    let id = self.next_id;
    self.next_id += 1;

    let cell = Cell { dimension, id };
    self.cells.insert(id, cell);

    // Add to lattice: element `id` is greater than its attachments `att_id`.
    self.attachment_lattice.add_element(id);
    for &att_id in attachments {
      self.attachment_lattice.add_relation(att_id, id); // att_id is a face of id
    }
    cell
  }

  /// Adds a new cell to the complex with specified attachments as [`Cell`] references.
  ///
  /// This is a convenience wrapper around [`CellComplex::add_cell_by_id`]. It extracts the IDs
  /// from the provided `attachments` slice of [`Cell`] references and then calls
  /// [`CellComplex::add_cell_by_id`].
  ///
  /// # Arguments
  ///
  /// * `dimension`: The dimension of the new cell.
  /// * `attachments`: A slice of references to [`Cell`] structs that are the maximal faces of the
  ///   new cell.
  ///
  /// # Returns
  ///
  /// The newly created [`Cell`].
  pub fn add_cell(&mut self, dimension: usize, attachments: &[&Cell]) -> Cell {
    let ids = attachments.iter().map(|cell| cell.id()).collect::<Vec<_>>();
    self.add_cell_by_id(dimension, &ids)
  }

  /// Retrieves a cell from the complex by its unique ID.
  ///
  /// # Arguments
  ///
  /// * `id`: The `usize` ID of the cell to retrieve.
  ///
  /// # Returns
  ///
  /// An `Option<Cell>` containing a copy of the [`Cell`] if found, otherwise `None`.
  pub fn get_cell(&self, id: usize) -> Option<Cell> { self.cells.get(&id).cloned() }

  /// Returns a vector of all cells in the complex having a specific `dimension`.
  ///
  /// The order of cells in the returned vector is not guaranteed.
  ///
  /// # Arguments
  ///
  /// * `dimension`: The dimension for which to retrieve cells.
  ///
  /// # Returns
  ///
  /// A `Vec<Cell>` containing copies of all cells of the specified dimension.
  pub fn cells_of_dimension(&self, dimension: usize) -> Vec<Cell> {
    self.cells.values().filter(|cell| cell.dimension() == dimension).cloned().collect()
  }

  /// Returns the maximum dimension of any cell currently in the complex.
  ///
  /// If the complex is empty, returns 0.
  ///
  /// # Returns
  ///
  /// The highest dimension among all cells in the complex.
  pub fn max_dimension(&self) -> usize {
    self.cells.values().map(Cell::dimension).max().unwrap_or(0)
  }

  /// Retrieves all cells that are directly attached to `cell` and have a higher dimension.
  ///
  /// These are the cells for which `cell` is a maximal face. In the context of the
  /// `attachment_lattice`, these are the direct successors of `cell.id()`.
  /// These are sometimes referred to as the cofaces that `cell` is a part of the boundary of.
  ///
  /// # Arguments
  ///
  /// * `cell`: A reference to the [`Cell`] whose higher-dimensional attachments (cofaces) are
  ///   sought.
  ///
  /// # Returns
  ///
  /// A `Vec<Cell>` of cells that have `cell` as one of their maximal faces.
  /// Panics if `cell.id()` is not found in the complex (should not happen if `cell` is from this
  /// complex).
  pub fn get_attachments(&self, cell: &Cell) -> Vec<Cell> {
    self
      .attachment_lattice
      .successors(cell.id()) // In lattice: cell.id < successor.id
      .into_iter()
      .map(|id| self.get_cell(id).expect("Successor cell ID not found in complex"))
      .collect()
  }

  /// Retrieves all maximal faces of the given `cell`.
  ///
  /// These are the cells of dimension `cell.dimension() - 1` to which `cell` is directly attached.
  /// In the context of the `attachment_lattice`, these are the direct predecessors of `cell.id()`.
  ///
  /// # Arguments
  ///
  /// * `cell`: A reference to the [`Cell`] whose maximal faces are sought.
  ///
  /// # Returns
  ///
  /// A `Vec<Cell>` of maximal faces of `cell`.
  /// Panics if `cell.id()` is not found in the complex or if a predecessor ID is invalid.
  pub fn faces(&self, cell: &Cell) -> Vec<Cell> {
    self
      .attachment_lattice
      .predecessors(cell.id()) // In lattice: predecessor.id < cell.id
      .into_iter()
      .map(|id| self.get_cell(id).expect("Predecessor cell ID not found in complex"))
      .collect()
  }
}

impl Collection for CellComplex {
  type Item = Cell;

  /// Checks if the complex contains the given `point` (cell).
  ///
  /// # Arguments
  ///
  /// * `point`: A reference to the [`Cell`] to check for containment.
  ///
  /// # Returns
  ///
  /// `true` if a cell with the same ID as `point.id()` exists in the complex, `false` otherwise.
  fn contains(&self, point: &Self::Item) -> bool { self.cells.contains_key(&point.id()) }

  /// Checks if the cell complex is empty (contains no cells).
  ///
  /// # Returns
  ///
  /// `true` if the `cells` map is empty, `false` otherwise.
  fn is_empty(&self) -> bool { self.cells.is_empty() }
}

/// Provides a partial order on cells based on the face/attachment relations.
///
/// The implementation delegates to the underlying `attachment_lattice` where cell IDs are used.
/// A cell `a` is considered less than or equal to cell `b` (`a <= b`) if `a` is a face of `b`
/// (or `a` is `b`).
impl Poset for CellComplex {
  /// Tests if cell `a` is a face of (or equal to) cell `b`.
  ///
  /// Delegates to `self.attachment_lattice.leq(&a.id, &b.id)`.
  ///
  /// # Arguments
  ///
  /// * `a`: A reference to the potential subcell (face).
  /// * `b`: A reference to the potential supercell.
  ///
  /// # Returns
  ///
  /// See [`Lattice::leq`]. `Some(true)` if `a` is a face of or equal to `b`.
  fn leq(&self, a: &Self::Item, b: &Self::Item) -> Option<bool> {
    self.attachment_lattice.leq(&a.id, &b.id)
  }

  /// Computes the upset of cell `a`: all cells that have `a` as a face (i.e., supercells of `a`).
  ///
  /// Delegates to `self.attachment_lattice.upset(a.id())` and maps IDs back to [`Cell`]s.
  /// Includes `a` itself if the lattice considers elements part of their own upsets.
  fn upset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .upset(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from upset not found"))
      .collect()
  }

  /// Computes the downset of cell `a`: all faces of `a`, including `a` itself.
  ///
  /// Delegates to `self.attachment_lattice.downset(a.id())` and maps IDs back to [`Cell`]s.
  fn downset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .downset(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from downset not found"))
      .collect()
  }

  /// Finds all minimal elements of the cell complex based on the attachment partial order.
  /// These are typically the 0-cells (vertices) or any cells that are not supercells of any other
  /// cell (e.g. isolated cells not part of any higher-dimensional cell's boundary).
  ///
  /// Delegates to `self.attachment_lattice.minimal_elements()`.
  fn minimal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .minimal_elements()
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from minimal_elements not found"))
      .collect()
  }

  /// Finds all maximal elements of the cell complex based on the attachment partial order.
  /// These are cells that are not faces of any other cell in the complex.
  ///
  /// Delegates to `self.attachment_lattice.maximal_elements()`.
  fn maximal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .maximal_elements()
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from maximal_elements not found"))
      .collect()
  }

  /// Computes the join (least upper bound) of two cells `a` and `b` in the attachment lattice.
  ///
  /// This corresponds to finding the smallest cell that has both `a` and `b` as faces, if such a
  /// unique cell exists. Delegates to `self.attachment_lattice.join(a.id(), b.id())`.
  fn join(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    self
      .attachment_lattice
      .join(a.id(), b.id())
      .map(|id| self.get_cell(id).expect("Cell ID from join not found"))
  }

  /// Computes the meet (greatest lower bound) of two cells `a` and `b` in the attachment lattice.
  ///
  /// This corresponds to finding the largest cell that is a face of both `a` and `b`, if such a
  /// unique cell exists. Delegates to `self.attachment_lattice.meet(a.id(), b.id())`.
  fn meet(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    self
      .attachment_lattice
      .meet(a.id(), b.id())
      .map(|id| self.get_cell(id).expect("Cell ID from meet not found"))
  }

  /// Finds all direct supercells (cofaces) of cell `a`.
  ///
  /// These are cells for which `a` is a maximal proper face. In the lattice, these are direct
  /// successors. This is equivalent to `self.get_attachments(&a)` if `get_attachments` is defined
  /// this way. Delegates to `self.attachment_lattice.successors(a.id())`.
  fn successors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .successors(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from successors not found"))
      .collect()
  }

  /// Finds all direct faces (maximal proper subcells) of cell `a`.
  ///
  /// These are the maximal cells that are faces of `a`. In the lattice, these are direct
  /// predecessors. This is equivalent to `self.faces(&a)`.
  /// Delegates to `self.attachment_lattice.predecessors(a.id())`.
  fn predecessors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .predecessors(a.id())
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from predecessors not found"))
      .collect()
  }
}

/// Defines topological operations for a `CellComplex`.
impl Topology for CellComplex {
  /// Returns the cofaces of `item`: cells of dimension `item.dimension() + 1` for which `item` is a
  /// maximal face.
  ///
  /// This interpretation of "neighborhood" corresponds to the cells in the open star of `item`
  /// that are immediately larger (i.e., direct successors in the face poset).
  /// It uses `self.attachment_lattice.successors` which aligns with the definition of
  /// `self.get_attachments()`.
  ///
  /// # Arguments
  ///
  /// * `item`: A reference to the [`Cell`] whose cofaces are sought.
  ///
  /// # Returns
  ///
  /// A `Vec<Cell>` containing the cofaces.
  fn neighborhood(&self, item: &Self::Item) -> Vec<Self::Item> {
    // This is equivalent to self.get_attachments(item) if attachments are direct cofaces.
    self
      .attachment_lattice
      .successors(item.id()) // Direct supercells in the face poset
      .into_iter()
      .map(|id| self.get_cell(id).expect("Cell ID from neighborhood (successors) not found"))
      .collect()
  }

  /// TODO: Implement this correctly.
  fn boundary<R: Ring + Copy>(&self, item: &Self::Item) -> Chain<Self, R> {
    let mut boundary_chain_items = Vec::new();
    let mut boundary_chain_coeffs = Vec::new();

    // `self.faces(item)` should return the maximal proper faces of `item`,
    // which are typically (item.dimension - 1)-cells.
    let faces = self.faces(item);
    for face in faces {
      // This coefficient logic is a placeholder for actual incidence numbers.
      // For a $(d-1)$-face of a $d$-cell, an incidence number is used.
      // The current logic just checks dimension and assigns +1/-1, which is not standard.
      let coeff = if face.dimension() == item.dimension() - 1 {
        R::one() // Placeholder: needs to be incidence number [item : face]
      } else {
        // This branch implies `face` is not a (d-1)-face, which would be unusual if `self.faces` is
        // correct. Or, it's a (d-1)-face but somehow the dimension check fails.
        // Standard boundary operator only sums (d-1)-faces.
        -R::one()
      };
      boundary_chain_items.push(face);
      boundary_chain_coeffs.push(coeff);
    }
    Chain::from_items_and_coeffs(self, boundary_chain_items, boundary_chain_coeffs)
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
    assert_eq!(complex.get_attachments(&e1), &[t1]);
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
