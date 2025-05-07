use std::collections::{BTreeSet, HashSet};

/// A trait for sets that support basic set operations.
///
/// This trait defines the fundamental operations that can be performed on sets:
/// containment testing, set difference, intersection, and union.
///
/// # Type Parameters
/// * `Point` - The type of elements contained in the set
pub trait Set {
  /// The type of elements contained in the set
  type Point;

  /// Tests if a point is contained in the set.
  ///
  /// # Arguments
  /// * `point` - The point to test for containment
  fn contains(&self, point: &Self::Point) -> bool;

  /// Computes the set difference (self - other).
  ///
  /// # Arguments
  /// * `other` - The set to subtract from this set
  fn difference(&self, other: &Self) -> Self;

  /// Computes the intersection of two sets.
  ///
  /// # Arguments
  /// * `other` - The set to intersect with this set
  fn intersect(&self, other: &Self) -> Self;

  /// Computes the union of two sets.
  ///
  /// # Arguments
  /// * `other` - The set to union with this set
  fn union(&self, other: &Self) -> Self;

  /// Tests if the set is empty.
  ///
  /// # Returns
  /// * `true` if the set is empty
  /// * `false` otherwise
  fn is_empty(&self) -> bool;
}

impl<T> Set for HashSet<T> {
  type Point = T;

  fn contains(&self, point: &Self::Point) -> bool { self.contains(point) }

  fn difference(&self, other: &Self) -> Self { self.difference(other).cloned().collect() }

  fn intersect(&self, other: &Self) -> Self { self.intersection(other).cloned().collect() }

  fn union(&self, other: &Self) -> Self { self.union(other).cloned().collect() }

  fn is_empty(&self) -> bool { self.is_empty() }
}

impl<T> Set for BTreeSet<T> {
  type Point = T;

  fn contains(&self, point: &Self::Point) -> bool { self.contains(point) }

  fn difference(&self, other: &Self) -> Self { self.difference(other).cloned().collect() }

  fn intersect(&self, other: &Self) -> Self { self.intersection(other).cloned().collect() }

  fn union(&self, other: &Self) -> Self { self.union(other).cloned().collect() }

  fn is_empty(&self) -> bool { self.is_empty() }
}
