//! Core traits and definitions for constructing a cell complex over arbitrary types.
//!
//! This module provides the fundamental building blocks for topological spaces and
//! CW complexes (cell complexes). It allows for the construction of spaces by
//! attaching cells of various dimensions along their boundaries.
//!
//! # Mathematical Background
//!
//! ## Topology
//!
//! A topology on a set X is a collection of subsets (called open sets) that satisfies:
//! - The empty set and X itself are open
//! - Arbitrary unions of open sets are open
//! - Finite intersections of open sets are open
//!
//! ## Cell Complex (CW Complex)
//!
//! A CW complex is built incrementally by:
//! - Starting with discrete points (0-cells)
//! - Attaching n-dimensional cells along their boundaries to the (n-1)-skeleton
//! - The n-skeleton consists of all cells of dimension ≤ n
//!
//! CW complexes provide a way to decompose topological spaces into simple building blocks:
//! - 0-cells: points
//! - 1-cells: line segments
//! - 2-cells: disks
//! - 3-cells: solid balls, etc.

use std::collections::HashSet;

use super::error::SpacesError;
use crate::definitions::{Set, TopologicalSpace};

/// `OpenSet` is a collection of `Point` equipped with `union` and `intersection` operations.
///
/// In topology, an open set is a fundamental concept that defines the structure of the space.
/// Open sets satisfy specific closure properties and determine which points are "near" each other.
pub trait OpenSet: IntoIterator<Item = Self::Point> + Clone + Set + Attach<Self> {
  /// builds an `OpenSet` from an `Iterator`
  fn from(iter: Box<dyn Iterator<Item = Self::Point>>) -> Self;
}

/// `Topology` is a collection of `OpenSet` intended to enforce closure over union and finite
/// intersection operations.
///
/// A topology defines the structure of a space by specifying which sets are considered "open".
/// It must satisfy specific axioms: empty set and the whole space are open, arbitrary unions
/// of open sets are open, and finite intersections of open sets are open.
pub trait Topology: TopologicalSpace {
  /// Type for points within `OpenSet`
  type Point;
  /// `OpenSet` implementation for this particular `Topology`
  type OpenSet: OpenSet<Point = <Self as Topology>::Point>;
  /// Return the collection of all `Point` in the topological space (the underlying set).
  fn points(&self) -> HashSet<<Self as Topology>::Point>;
  /// Return the collection of neighborhoods (open sets) already constructed containing the given
  /// point.
  ///
  /// In topology, a neighborhood of a point is an open set containing that point.
  /// This function returns all such open sets for the given point.
  fn neighborhoods(&self, point: <Self as Topology>::Point)
    -> HashSet<<Self as Topology>::OpenSet>;
}

/// Trait for defining the particular attachment map provided an openset and skeleton
pub trait Attach<O: OpenSet> {
  /// Attachment map implementation for this particular type of k-cell.
  ///
  /// The attachment map defines how a cell is glued to the existing skeleton.
  /// It maps points from the boundary of the cell to points in the (k-1)-skeleton.
  fn attach_boundary(&self, skeleton: &Skeleton<O>) -> O;
}

/// A k-cell of arbitrary type and dimension
pub struct Cell<O: OpenSet> {
  /// The openset that the k-cell consists of.
  pub cell: O,
  /// `k`, the dimension of the k-cell
  pub dimension: usize,
  /// Collection of incident cell IDs within the cell complex.
  ///
  /// Incident cells are those that share boundary components with this cell.
  /// For example, a 1-cell (edge) is incident to its endpoint 0-cells (vertices),
  /// and possibly to 2-cells (faces) that have this edge in their boundary.
  pub incidents: HashSet<(usize, usize)>,
  /// Boundary set of the k-cell
  pub boundary_set: O,
  /// The set of corresponding points the boundary is mapped to within
  /// a particular cell complex the cell attaches to
  pub boundary_set_correspondence: Option<O>,
}

impl<O: OpenSet> Cell<O> {
  /// Remove the boundary points from the core `cell` field if necessary
  pub fn remove_boundary(&mut self) { self.cell = self.cell.difference(&self.boundary_set); }

  /// Generate a new k-cell
  pub fn new(cell: O, boundary_set: O, dimension: usize) -> Self {
    Self {
      cell,
      dimension,
      incidents: HashSet::new(),
      boundary_set,
      boundary_set_correspondence: None,
    }
  }

  fn attach(&mut self, skeleton: &mut Skeleton<O>) {
    let boundary_set_correspondence = self.boundary_set.attach_boundary(skeleton);
    let max = skeleton.cells[self.dimension].len();
    for point in boundary_set_correspondence {
      let mut change: Option<(usize, usize)> = None;
      skeleton.cells.iter().enumerate().for_each(|(i, x)| {
        x.iter().enumerate().for_each(|(j, y)| {
          if y.cell.contains(&point) {
            self.incidents.insert((i, j));
            change = Some((i, j));
          }
        })
      });
      if let Some((i, j)) = change {
        skeleton.cells[i][j].incidents.insert((self.dimension, max));
      }
    }
  }
}

/// A `Skeleton` is a collection of `Cells` that have been glued together along `Cell::attach` maps.
///
/// In CW complex terminology, the n-skeleton consists of all cells of dimension ≤ n.
/// Building a CW complex involves constructing successive skeletons by attaching cells
/// of increasing dimension.
pub struct Skeleton<O: OpenSet> {
  /// Dimension of the `Skeleton` (maximum cell dimension contained)
  pub dimension: usize,
  /// The collection of `Cells` forming the skeleton with [dimension][idx]
  pub cells:     Vec<Vec<Cell<O>>>,
}

impl<O: OpenSet> Skeleton<O> {
  /// Initialize a new `Skeleton`
  pub fn init() -> Self { Self { dimension: 0, cells: vec![Vec::new()] } }

  /// Attach a cell to the existing cell complex.
  ///
  /// This implements the core operation in building a CW complex: attaching new cells
  /// to the existing skeleton. The process involves:
  /// 1. Verifying the dimensional constraints (can only attach n-cells to (n-1)-skeleton)
  /// 2. Finding boundary points and their images under the attachment map
  /// 3. Updating incidence relationships between cells
  /// 4. Updating the skeleton's dimension if needed
  pub fn attach(&mut self, mut cell: Cell<O>) -> Result<(), SpacesError> {
    let incoming_dim = cell.dimension as i64;
    if incoming_dim - self.dimension as i64 > 1 {
      return Err(SpacesError::DimensionMismatch);
    }
    cell.attach(self);
    if cell.dimension < self.cells.len() {
      self.cells[cell.dimension].push(cell);
    }
    Ok(())
  }

  /// Fetches the cell information containing a particular `Point`.
  ///
  /// This operation allows for point-based lookup within the cell complex,
  /// which is useful for navigating the topological structure.
  pub fn fetch_cell_by_point(&self, point: O::Point) -> Result<(&Cell<O>, usize), SpacesError> {
    for i in 0..self.cells.len() {
      for j in 0..self.cells[i].len() {
        if self.cells[i][j].cell.contains(&point) {
          return Ok((&self.cells[i][j], i));
        };
      }
    }
    Err(SpacesError::NoPointFound)
  }

  /// Returns the collection of all incident cells to the cell at index `cell_idx`.
  ///
  /// Incidence relationships capture how cells of different dimensions are connected.
  /// This is essential for algorithms that need to traverse the cell complex.
  pub fn incident_cells(
    &self,
    cell_idx: usize,
    cell_dimension: usize,
  ) -> Result<&HashSet<(usize, usize)>, SpacesError> {
    if cell_dimension >= self.cells.len() {
      return Err(SpacesError::DimensionMismatch);
    }
    if cell_idx >= self.cells[cell_dimension].len() {
      return Err(SpacesError::InvalidCellIdx);
    }
    Ok(&self.cells[cell_dimension][cell_idx].incidents)
  }

  /// Returns the collection of incident cells to `cell_idx` with exactly 1 dimension difference.
  ///
  /// This separates boundary relationships (cells of dimension k-1, forming the boundary)
  /// from coboundary relationships (cells of dimension k+1, having this cell in their boundary).
  /// This distinction is important for homology and cohomology calculations.
  pub fn filter_incident_by_dim(
    &self,
    cell_idx: usize,
    cell_dimension: usize,
  ) -> Result<HashSet<(usize, usize)>, SpacesError> {
    if cell_dimension >= self.cells.len() {
      return Err(SpacesError::DimensionMismatch);
    }
    if cell_idx >= self.cells[cell_dimension].len() {
      return Err(SpacesError::InvalidCellIdx);
    }
    let incidents = &self.cells[cell_dimension][cell_idx].incidents;
    let mut lower = HashSet::new();
    for (i, j) in incidents {
      if (cell_dimension as i64 - self.cells[*i][*j].dimension as i64).abs() == 1 {
        lower.insert((*i, *j));
      }
    }
    Ok(lower)
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use super::{Attach, Cell, OpenSet, Set, Skeleton, SpacesError};

  /// A trivial point type.
  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  struct TestPoint(&'static str);

  /// A super‐minimal OpenSet impl wrapping a HashSet<TestPoint>.
  #[derive(Debug, Clone, PartialEq, Eq)]
  struct TestOpenSet {
    points: HashSet<TestPoint>,
  }

  impl TestOpenSet {
    fn new<I: IntoIterator<Item = TestPoint>>(pts: I) -> Self {
      Self { points: pts.into_iter().collect() }
    }
  }

  // Bring in the Set trait you imported at file scope.
  impl Set for TestOpenSet {
    type Point = TestPoint;

    fn union(&self, other: &Self) -> Self {
      let mut pts = self.points.clone();
      pts.extend(other.points.iter().cloned());
      Self { points: pts }
    }

    fn intersect(&self, other: &Self) -> Self {
      let pts = self.points.intersection(&other.points).cloned().collect();
      Self { points: pts }
    }

    fn contains(&self, pt: &Self::Point) -> bool { self.points.contains(pt) }

    fn difference(&self, other: &Self) -> Self {
      let pts = self.points.difference(&other.points).cloned().collect();
      Self { points: pts }
    }

    fn is_empty(&self) -> bool { self.points.is_empty() }
  }

  impl IntoIterator for TestOpenSet {
    type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;
    type Item = TestPoint;

    fn into_iter(self) -> Self::IntoIter { self.points.into_iter() }
  }

  impl OpenSet for TestOpenSet {
    fn from(iter: Box<dyn Iterator<Item = TestPoint>>) -> Self {
      let pts = iter.collect();
      Self { points: pts }
    }
  }

  // A no-op attach: just returns the same set.
  impl Attach<TestOpenSet> for TestOpenSet {
    fn attach_boundary(&self, _skeleton: &Skeleton<TestOpenSet>) -> TestOpenSet { self.clone() }
  }

  #[test]
  fn test_skeleton_init() {
    let sk: Skeleton<TestOpenSet> = Skeleton::init();
    // dimension stays 0, and we have exactly one Vec<Cell> slot.
    assert_eq!(sk.dimension, 0);
    assert_eq!(sk.cells.len(), 1);
    assert!(sk.cells[0].is_empty());
  }

  #[test]
  fn test_attach_vertex() {
    let mut sk: Skeleton<TestOpenSet> = Skeleton::init();
    let pt = TestPoint("A");
    let cell_oset = TestOpenSet::new(vec![pt.clone()]);
    let boundary = TestOpenSet::new(vec![]);

    let c = Cell::new(cell_oset.clone(), boundary.clone(), 0);
    assert!(sk.attach(c).is_ok());

    // We should now have exactly one 0-cell in sk.cells[0].
    assert_eq!(sk.cells[0].len(), 1);
    assert!(sk.cells[0][0].cell.contains(&pt));
  }

  #[test]
  fn test_remove_boundary() {
    // A cell whose cell={A,B} and boundary={A} should drop A
    let mut c = Cell::new(
      TestOpenSet::new(vec![TestPoint("A"), TestPoint("B")]),
      TestOpenSet::new(vec![TestPoint("A")]),
      0,
    );
    c.remove_boundary();
    let left: HashSet<_> = c.cell.into_iter().collect();
    assert_eq!(left, vec![TestPoint("B")].into_iter().collect());
  }

  #[test]
  fn test_dimension_mismatch() {
    let mut sk: Skeleton<TestOpenSet> = Skeleton::init();
    // attaching a 2-cell onto a 0-skeleton should error
    let bad = Cell::new(TestOpenSet::new(vec![]), TestOpenSet::new(vec![]), 2);
    assert!(matches!(sk.attach(bad), Err(SpacesError::DimensionMismatch)));
  }

  #[test]
  fn test_fetch_cell_by_point() {
    let mut sk: Skeleton<TestOpenSet> = Skeleton::init();
    let pt = TestPoint("A");
    sk.attach(Cell::new(TestOpenSet::new(vec![pt.clone()]), TestOpenSet::new(vec![]), 0)).unwrap();

    let found = sk.fetch_cell_by_point(pt.clone());
    assert!(found.is_ok());
    let (c_ref, dim) = found.unwrap();
    assert_eq!(dim, 0);
    assert!(c_ref.cell.contains(&pt));

    // Something I never added should error.
    assert!(matches!(sk.fetch_cell_by_point(TestPoint("Z")), Err(SpacesError::NoPointFound)));
  }

  #[test]
  fn test_incident_and_filter_errors() {
    let sk: Skeleton<TestOpenSet> = Skeleton::init();
    // Asking for a nonexistent dimension should be a DimensionMismatch
    assert!(matches!(sk.incident_cells(0, 1), Err(SpacesError::DimensionMismatch)));
    assert!(matches!(sk.filter_incident_by_dim(0, 1), Err(SpacesError::DimensionMismatch)));

    // Bad index within valid dim should be InvalidCellIdx
    assert!(matches!(sk.incident_cells(99, 0), Err(SpacesError::InvalidCellIdx)));
  }
}
