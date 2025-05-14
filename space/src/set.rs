//! A module providing a generic trait for set operations.
//!
//! This module defines the `Set` trait which abstracts over different set implementations
//! (like `HashSet` and `BTreeSet`) and provides a common interface for basic set operations:
//! - Containment testing
//! - Set difference
//! - Intersection
//! - Union
//! - Emptyness testing
//!
//! # Implementations
//!
//! The trait is implemented for:
//! - `HashSet<T, S>` where `T: Hash + Eq + Clone` and `S: BuildHasher + Default`
//! - `BTreeSet<T>` where `T: Ord + Clone`
//!
//! # Example
//! ```rust
//! use std::collections::HashSet;
//!
//! use harness_space::set::Set;
//!
//! let a: HashSet<_> = [1, 2, 3].into_iter().collect();
//! let b: HashSet<_> = [2, 3, 4].into_iter().collect();
//!
//! let intersection = a.meet(&b);
//! assert!(intersection.contains(&2));
//! assert!(intersection.contains(&3));
//! ```

use std::{
  collections::{BTreeSet, HashSet},
  hash::{BuildHasher, Hash},
};

pub trait Collection {
  /// The type of elements contained in the collection
  type Point;

  /// Tests if a point is contained in the collection.
  ///
  /// # Arguments
  /// * `point` - The point to test for containment
  fn contains(&self, point: &Self::Point) -> bool;

  /// Tests if the set is empty.
  ///
  /// # Returns
  /// * `true` if the set is empty
  /// * `false` otherwise
  fn is_empty(&self) -> bool;
}

/// A trait for sets that support basic set operations.
///
/// This trait defines the fundamental operations that can be performed on sets:
/// containment testing, set difference, intersection, and union.
///
/// # Type Parameters
/// * `Point` - The type of elements contained in the set
pub trait Set: Collection {
  /// Computes the set difference (self - other).
  ///
  /// # Arguments
  /// * `other` - The set to subtract from this set
  fn minus(&self, other: &Self) -> Self;

  /// Computes the intersection (meet) of two sets.
  ///
  /// # Arguments
  /// * `other` - The set to intersect with this set
  fn meet(&self, other: &Self) -> Self;

  /// Computes the union (join) of two sets.
  ///
  /// # Arguments
  /// * `other` - The set to union with this set
  fn join(&self, other: &Self) -> Self;
}

/// A trait for sets that support partial order relations.
///
/// This trait extends the `Set` trait with a method for checking if one point is less than or equal
/// to another.
///
/// # Type Parameters
/// * `Point` - The type of elements contained in the set
pub trait Poset: Set {
  /// Tests if one point is less than or equal to another.
  ///
  /// # Arguments
  /// * `a` - The first point
  /// * `b` - The second point
  ///
  /// # Returns
  /// * `Some(true)` if `a` is less than or equal to `b`
  fn leq(&self, a: &Self::Point, b: &Self::Point) -> Option<bool>;
}

impl<T: Hash + Eq + Clone, S: BuildHasher + Default> Collection for HashSet<T, S> {
  type Point = T;

  fn contains(&self, point: &Self::Point) -> bool { Self::contains(self, point) }

  fn is_empty(&self) -> bool { Self::is_empty(self) }
}

impl<T: Hash + Eq + Clone, S: BuildHasher + Default> Set for HashSet<T, S> {
  fn minus(&self, other: &Self) -> Self { Self::difference(self, other).cloned().collect() }

  fn meet(&self, other: &Self) -> Self { Self::intersection(self, other).cloned().collect() }

  fn join(&self, other: &Self) -> Self { Self::union(self, other).cloned().collect() }
}

impl<T: Ord + Clone> Collection for BTreeSet<T> {
  type Point = T;

  fn contains(&self, point: &Self::Point) -> bool { Self::contains(self, point) }

  fn is_empty(&self) -> bool { Self::is_empty(self) }
}

impl<T: Ord + Clone> Set for BTreeSet<T> {
  fn minus(&self, other: &Self) -> Self { Self::difference(self, other).cloned().collect() }

  fn meet(&self, other: &Self) -> Self { Self::intersection(self, other).cloned().collect() }

  fn join(&self, other: &Self) -> Self { Self::union(self, other).cloned().collect() }
}

impl<T: Ord + Clone> Poset for BTreeSet<T> {
  fn leq(&self, a: &Self::Point, b: &Self::Point) -> Option<bool> {
    if self.contains(a) && self.contains(b) {
      Some(a <= b)
    } else {
      None
    }
  }
}

impl<T: PartialEq> Collection for Vec<T> {
  type Point = T;

  fn contains(&self, point: &Self::Point) -> bool { self.iter().any(|p| p == point) }

  fn is_empty(&self) -> bool { self.is_empty() }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use super::*;

  #[test]
  fn test_set_operations() {
    let a: HashSet<_> = [1, 2, 3].into_iter().collect();
    let b: HashSet<_> = [2, 3, 4].into_iter().collect();

    let intersection = a.meet(&b);
    assert_eq!(intersection.len(), 2);
    assert!(intersection.contains(&2));
    assert!(intersection.contains(&3));

    let union = a.join(&b);
    assert_eq!(union.len(), 4);
    assert!(union.contains(&1));
    assert!(union.contains(&2));
    assert!(union.contains(&3));
    assert!(union.contains(&4));

    let difference = a.minus(&b);
    assert_eq!(difference.len(), 1);
    assert!(difference.contains(&1));

    assert!(!a.is_empty());
    assert!(!b.is_empty());
  }
}
