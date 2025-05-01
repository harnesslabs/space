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

use std::{collections::HashSet, hash::Hash};

use super::error::SpacesError;
use crate::definitions::{Set, TopologicalSpace};

/// `Point` wrapper of a generic type `T`.
///
/// Represents a 0-dimensional point in the topological space.
/// These form the building blocks of higher dimensional structures.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Point<T: Eq + std::hash::Hash + Clone + Sized>(T);

/// `OpenSet` is a collection of `Point` equipped with `union` and `intersection` operations.
///
/// In topology, an open set is a fundamental concept that defines the structure of the space.
/// Open sets satisfy specific closure properties and determine which points are "near" each other.
pub trait OpenSet: IntoIterator<Item = <Self as OpenSet>::Point> + Clone + Set {
  /// `Point` type within the set
  type Point: Eq + std::hash::Hash;
  /// builds an `OpenSet` from an `Iterator`
  fn from(iter: Box<dyn Iterator<Item = <Self as OpenSet>::Point>>) -> Self;
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

/// Trait for generic k-cell implementation over type `T`.
///
/// In a cell complex, a k-cell is homeomorphic to an open k-dimensional disk.
/// For example:
/// - 0-cell: a point
/// - 1-cell: an open interval (no endpoints)
/// - 2-cell: an open disk (no boundary circle)
/// - 3-cell: an open ball (no boundary sphere)
///
/// Cells are attached to the existing complex along their boundaries.
pub trait KCell<T: Eq + Hash + Clone, O: OpenSet> {
  /// Returns the collection of all `Point` in the k-cell
  fn points(&self) -> HashSet<&<O as OpenSet>::Point>;
  /// Returns the dimension k
  fn dimension(&self) -> usize;
  /// Returns the set of boundary points of the `KCell`.
  ///
  /// The boundary of a k-cell is a collection of (k-1) dimensional cells.
  /// For example, the boundary of a 2-cell (disk) is a 1-dimensional circle.
  fn boundary(&mut self) -> HashSet<<O as OpenSet>::Point>;
  /// Attachment map implementation for this particular type of k-cell.
  ///
  /// The attachment map defines how a cell is glued to the existing skeleton.
  /// It maps points from the boundary of the cell to points in the (k-1)-skeleton.
  fn attach(
    &self,
    point: &<O as OpenSet>::Point,
    skeleton: &mut Skeleton<T, O>,
  ) -> <O as OpenSet>::Point;
  /// Remove points from the k-cell.
  ///
  /// Used when refining the cell structure, such as when removing boundary points
  /// after attaching a cell to the skeleton.
  fn remove(&mut self, set: HashSet<<O as OpenSet>::Point>) -> bool;
}

/// `KCell` wrapper struct with incidence information.
///
/// Maintains the cell along with its relationships to other cells in the complex.
/// The incidence relationship is essential for traversing the cell complex structure.
pub struct Cell<T: Eq + Hash + Clone, O: OpenSet> {
  /// Smart pointer to the `KCell`
  pub cell:      Box<dyn KCell<T, O>>,
  /// Collection of incident cell IDs within the cell complex.
  ///
  /// Incident cells are those that share boundary components with this cell.
  /// For example, a 1-cell (edge) is incident to its endpoint 0-cells (vertices),
  /// and possibly to 2-cells (faces) that have this edge in their boundary.
  pub incidents: Vec<usize>,
}

impl<T: Eq + Hash + Clone, O: OpenSet> Cell<T, O> {
  /// Create a new `Cell` from a `KCell`
  pub fn new(cell: Box<dyn KCell<T, O>>) -> Self { Self { cell, incidents: Vec::new() } }
}

/// A `Skeleton` is a collection of `Cells` that have been glued together along `Cell::attach` maps.
///
/// In CW complex terminology, the n-skeleton consists of all cells of dimension ≤ n.
/// Building a CW complex involves constructing successive skeletons by attaching cells
/// of increasing dimension.
pub struct Skeleton<T: Eq + Hash + Clone, O: OpenSet> {
  /// Dimension of the `Skeleton` (maximum cell dimension contained)
  pub dimension: usize,
  /// The collection of `Cells` forming the skeleton
  pub cells:     Vec<Cell<T, O>>,
}

impl<T: Eq + Hash + Clone, O: OpenSet> Skeleton<T, O> {
  /// Initialize a new `Skeleton`
  pub fn init() -> Self { Self { dimension: 0, cells: Vec::new() } }

  /// Attach a cell to the existing cell complex.
  ///
  /// This implements the core operation in building a CW complex: attaching new cells
  /// to the existing skeleton. The process involves:
  /// 1. Verifying the dimensional constraints (can only attach n-cells to (n-1)-skeleton)
  /// 2. Finding boundary points and their images under the attachment map
  /// 3. Updating incidence relationships between cells
  /// 4. Updating the skeleton's dimension if needed
  pub fn attach(&mut self, cell: Box<dyn KCell<T, O>>) -> Result<(), SpacesError> {
    let incoming_dim = cell.dimension() as i64;
    if incoming_dim - self.dimension as i64 > 1 {
      return Err(SpacesError::DimensionMismatch);
    }
    if self.dimension == 0 && incoming_dim == 1 && self.cells.is_empty() {
      return Err(SpacesError::CWUninitialized);
    }
    let mut cell = Cell::new(cell);
    let mut boundary = cell.cell.boundary();
    let mut incident_indices = Vec::new();
    for p in cell.cell.points() {
      let point = cell.cell.attach(p, self);
      let mut truth = false;
      self.cells.iter().enumerate().for_each(|(i, x)| {
        if x.cell.points().contains(&point) {
          truth = true;
          if !cell.incidents.contains(&i) {
            cell.incidents.push(i);
            incident_indices.push(i);
          }
        }
      });
      if truth {
        boundary.insert(point);
      }
    }
    let new_cell_idx = self.cells.len();
    cell.cell.remove(boundary);
    self.cells.push(cell);
    for idx in incident_indices {
      if !self.cells[idx].incidents.contains(&new_cell_idx) {
        self.cells[idx].incidents.push(new_cell_idx);
      }
    }
    if incoming_dim - self.dimension as i64 == 1 {
      self.dimension += 1
    }
    Ok(())
  }

  /// Fetches the cell information containing a particular `Point`.
  ///
  /// This operation allows for point-based lookup within the cell complex,
  /// which is useful for navigating the topological structure.
  pub fn fetch_cell_by_point(
    &self,
    point: <O as OpenSet>::Point,
  ) -> Result<(&Cell<T, O>, usize), SpacesError> {
    for i in 0..self.cells.len() {
      if self.cells[i].cell.points().contains(&point) {
        return Ok((&self.cells[i], i));
      };
    }
    Err(SpacesError::NoPointFound)
  }

  /// Returns the collection of all incident cells to the cell at index `cell_idx`.
  ///
  /// Incidence relationships capture how cells of different dimensions are connected.
  /// This is essential for algorithms that need to traverse the cell complex.
  pub fn incident_cells(&self, cell_idx: usize) -> Result<&Vec<usize>, SpacesError> {
    if cell_idx >= self.cells.len() {
      return Err(SpacesError::InvalidCellIdx);
    }
    Ok(&self.cells[cell_idx].incidents)
  }

  /// Returns the collection of incident cells to `cell_idx` with exactly 1 dimension difference.
  ///
  /// This separates boundary relationships (cells of dimension k-1, forming the boundary)
  /// from coboundary relationships (cells of dimension k+1, having this cell in their boundary).
  /// This distinction is important for homology and cohomology calculations.
  pub fn filter_incident_by_dim(
    &self,
    cell_idx: usize,
  ) -> Result<(Vec<usize>, Vec<usize>), SpacesError> {
    if cell_idx >= self.cells.len() {
      return Err(SpacesError::InvalidCellIdx);
    }
    let incidents = &self.cells[cell_idx].incidents;
    let dim = self.cells[cell_idx].cell.dimension();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for i in incidents {
      if dim as i64 - self.cells[*i].cell.dimension() as i64 == 1 {
        lower.push(*i);
      } else if self.cells[*i].cell.dimension() as i64 - dim as i64 == 1 {
        upper.push(*i);
      }
    }
    Ok((lower, upper))
  }
}

#[cfg(test)]
mod tests {
  use std::{collections::HashSet, hash::Hash};

  use super::*;

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  struct TestPoint(String);

  #[derive(Debug, Clone, PartialEq, Eq)]
  struct TestOpenSet {
    points: HashSet<TestPoint>,
  }

  impl OpenSet for TestOpenSet {
    type Point = TestPoint;

    fn from(iter: Box<dyn Iterator<Item = <Self as OpenSet>::Point>>) -> Self {
      let points = iter.collect();
      Self { points }
    }
  }

  impl Set for TestOpenSet {
    type Point = TestPoint;

    fn union(&self, other: &TestOpenSet) -> Self {
      let mut points = self.points.clone();
      for point in other.points.iter() {
        points.insert(point.clone());
      }
      Self { points }
    }

    fn intersect(&self, other: &TestOpenSet) -> Self {
      let mut points = HashSet::new();
      for point in self.points.iter() {
        if other.points.contains(point) {
          points.insert(point.clone());
        }
      }
      Self { points }
    }

    fn contains(&self, point: &Self::Point) -> bool { self.points.contains(point) }

    fn difference(&self, other: &Self) -> Self {
      let this = self.points.difference(&other.points).cloned().collect::<HashSet<_>>();
      TestOpenSet { points: this }
    }
  }

  impl IntoIterator for TestOpenSet {
    type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;
    type Item = TestPoint;

    fn into_iter(self) -> Self::IntoIter { self.points.into_iter() }
  }

  struct TestVertex {
    point: TestPoint,
  }

  impl KCell<String, TestOpenSet> for TestVertex {
    fn points(&self) -> HashSet<&TestPoint> {
      let mut points = HashSet::new();
      points.insert(&self.point);
      points
    }

    fn dimension(&self) -> usize { 0 }

    fn boundary(&mut self) -> HashSet<TestPoint> { HashSet::new() }

    fn attach(
      &self,
      _point: &TestPoint,
      _skeleton: &mut Skeleton<String, TestOpenSet>,
    ) -> TestPoint {
      self.point.clone()
    }

    fn remove(&mut self, _set: HashSet<TestPoint>) -> bool { false }
  }

  struct TestEdge {
    start:   TestPoint,
    end:     TestPoint,
    removed: HashSet<TestPoint>,
  }

  impl KCell<String, TestOpenSet> for TestEdge {
    fn points(&self) -> HashSet<&TestPoint> {
      let mut points = HashSet::new();
      points.insert(&self.start);
      points.insert(&self.end);
      points
    }

    fn dimension(&self) -> usize { 1 }

    fn boundary(&mut self) -> HashSet<TestPoint> {
      let mut boundary = HashSet::new();
      boundary.insert(self.start.clone());
      boundary.insert(self.end.clone());
      boundary
    }

    fn attach(
      &self,
      point: &TestPoint,
      _skeleton: &mut Skeleton<String, TestOpenSet>,
    ) -> TestPoint {
      if point == &self.start {
        self.start.clone()
      } else {
        self.end.clone()
      }
    }

    fn remove(&mut self, set: HashSet<TestPoint>) -> bool {
      self.removed = set;
      !self.removed.is_empty()
    }
  }

  struct TestFace {
    vertices: Vec<TestPoint>,
    _edges:   Vec<String>,
    _id:      String,
    removed:  HashSet<TestPoint>,
  }

  impl KCell<String, TestOpenSet> for TestFace {
    fn points(&self) -> HashSet<&TestPoint> {
      let mut points = HashSet::new();
      for v in &self.vertices {
        points.insert(v);
      }
      points
    }

    fn dimension(&self) -> usize { 2 }

    fn boundary(&mut self) -> HashSet<TestPoint> {
      let mut boundary = HashSet::new();
      for v in &self.vertices {
        boundary.insert(v.clone());
      }
      boundary
    }

    fn attach(
      &self,
      point: &TestPoint,
      _skeleton: &mut Skeleton<String, TestOpenSet>,
    ) -> TestPoint {
      for v in &self.vertices {
        if v == point {
          return v.clone();
        }
      }
      self.vertices[0].clone() // Default to first vertex if not found
    }

    fn remove(&mut self, set: HashSet<TestPoint>) -> bool {
      self.removed = set;
      !self.removed.is_empty()
    }
  }

  #[test]
  fn test_skeleton_init() {
    let skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    assert_eq!(skeleton.dimension, 0);
    assert_eq!(skeleton.cells.len(), 0);
  }

  #[test]
  fn test_skeleton_attach_vertex() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    let vertex = TestVertex { point: TestPoint("A".to_string()) };

    let result = skeleton.attach(Box::new(vertex));
    assert!(result.is_ok());
    assert_eq!(skeleton.dimension, 0);
    assert_eq!(skeleton.cells.len(), 1);
  }

  #[test]
  fn test_skeleton_attach_edge() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    let vertex1 = TestVertex { point: TestPoint("A".to_string()) };
    let vertex2 = TestVertex { point: TestPoint("B".to_string()) };
    skeleton.attach(Box::new(vertex1)).unwrap();
    skeleton.attach(Box::new(vertex2)).unwrap();

    let edge = TestEdge {
      start:   TestPoint("A".to_string()),
      end:     TestPoint("B".to_string()),
      removed: HashSet::new(),
    };

    let result = skeleton.attach(Box::new(edge));
    assert!(result.is_ok());
    assert_eq!(skeleton.dimension, 1);
    assert_eq!(skeleton.cells.len(), 3);
  }

  #[test]
  fn test_skeleton_dimension_error() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();

    let face = TestFace {
      vertices: vec![
        TestPoint("A".to_string()),
        TestPoint("B".to_string()),
        TestPoint("C".to_string()),
      ],
      _edges:   vec!["Edge_AB".to_string(), "Edge_BC".to_string(), "Edge_CA".to_string()],
      _id:      "Face_ABC".to_string(),
      removed:  HashSet::new(),
    };

    let result = skeleton.attach(Box::new(face));
    assert!(matches!(result, Err(SpacesError::DimensionMismatch)));
  }

  #[test]
  fn test_skeleton_uninitialized_error() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();

    let edge = TestEdge {
      start:   TestPoint("A".to_string()),
      end:     TestPoint("B".to_string()),
      removed: HashSet::new(),
    };

    let result = skeleton.attach(Box::new(edge));
    assert!(matches!(result, Err(SpacesError::CWUninitialized)));
  }

  #[test]
  fn test_skeleton_fetch_cell_by_point() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    let vertex = TestVertex { point: TestPoint("A".to_string()) };
    skeleton.attach(Box::new(vertex)).unwrap();

    let result = skeleton.fetch_cell_by_point(TestPoint("A".to_string()));
    assert!(result.is_ok());
    let result = skeleton.fetch_cell_by_point(TestPoint("Z".to_string()));
    assert!(matches!(result, Err(SpacesError::NoPointFound)));
  }

  #[test]
  fn test_skeleton_incident_cells() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    let vertex1 = TestVertex { point: TestPoint("A".to_string()) };
    let vertex2 = TestVertex { point: TestPoint("B".to_string()) };

    skeleton.attach(Box::new(vertex1)).unwrap();
    skeleton.attach(Box::new(vertex2)).unwrap();

    let edge = TestEdge {
      start:   TestPoint("A".to_string()),
      end:     TestPoint("B".to_string()),
      removed: HashSet::new(),
    };
    skeleton.attach(Box::new(edge)).unwrap();

    let result = skeleton.incident_cells(0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
    let result = skeleton.incident_cells(1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
    let result = skeleton.incident_cells(2);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
    let result = skeleton.incident_cells(99);
    assert!(matches!(result, Err(SpacesError::InvalidCellIdx)));
  }

  #[test]
  fn test_skeleton_filter_incident_by_dim() {
    let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
    let vertex1 = TestVertex { point: TestPoint("A".to_string()) };
    let vertex2 = TestVertex { point: TestPoint("B".to_string()) };
    let vertex3 = TestVertex { point: TestPoint("C".to_string()) };

    skeleton.attach(Box::new(vertex1)).unwrap();
    skeleton.attach(Box::new(vertex2)).unwrap();
    skeleton.attach(Box::new(vertex3)).unwrap();

    let edge1 = TestEdge {
      start:   TestPoint("A".to_string()),
      end:     TestPoint("B".to_string()),
      removed: HashSet::new(),
    };
    let edge2 = TestEdge {
      start:   TestPoint("B".to_string()),
      end:     TestPoint("C".to_string()),
      removed: HashSet::new(),
    };
    let edge3 = TestEdge {
      start:   TestPoint("C".to_string()),
      end:     TestPoint("A".to_string()),
      removed: HashSet::new(),
    };
    skeleton.attach(Box::new(edge1)).unwrap();
    skeleton.attach(Box::new(edge2)).unwrap();
    skeleton.attach(Box::new(edge3)).unwrap();

    let face = TestFace {
      vertices: vec![
        TestPoint("A".to_string()),
        TestPoint("B".to_string()),
        TestPoint("C".to_string()),
      ],
      _edges:   vec!["Edge_AB".to_string(), "Edge_BC".to_string(), "Edge_CA".to_string()],
      _id:      "Face_ABC".to_string(),
      removed:  HashSet::new(),
    };
    skeleton.attach(Box::new(face)).unwrap();

    let result = skeleton.filter_incident_by_dim(0);
    assert!(result.is_ok());
    let (lower, upper) = result.unwrap();
    assert_eq!(lower.len(), 0);
    assert_eq!(upper.len(), 2);
    let result = skeleton.filter_incident_by_dim(3);
    assert!(result.is_ok());
    let (lower, upper) = result.unwrap();
    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 1);
    let result = skeleton.filter_incident_by_dim(6);
    assert!(result.is_ok());
    let (lower, upper) = result.unwrap();
    assert_eq!(lower.len(), 3);
    assert_eq!(upper.len(), 0);
  }
}
