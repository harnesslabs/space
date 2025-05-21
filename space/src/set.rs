//! # Set Module
//!
//! This module provides traits for abstracting over collection types and partially
//! ordered sets (posets).
//!
//! The core traits defined are:
//! - [`Collection`]: For types that represent a collection of items, supporting basic operations
//!   like checking for containment and emptiness.
//! - [`Poset`]: Extends [`Collection`] for sets that also define a partial order relation among
//!   their items, along with related poset operations like finding upsets, downsets,
//!   minimal/maximal elements, etc.
//!
//! Implementations of [`Collection`] are provided for standard library types such as
//! [`HashSet`], [`BTreeSet`], and [`Vec`].

use std::{
  collections::{BTreeSet, HashSet},
  hash::{BuildHasher, Hash},
};

/// A trait for collections that support basic set operations.
///
/// This trait defines fundamental operations applicable to various collection types,
/// focusing on item containment and checking for emptiness.
///
/// # Type Parameters
///
/// * `Item`: The type of elements contained within the collection.
///
/// # Implementations
///
/// Implementations are provided for:
/// * [`HashSet<T, S>`]: Where `T: Hash + Eq + Clone` and `S: BuildHasher + Default`.
/// * [`BTreeSet<T>`]: Where `T: Ord + Clone`.
/// * [`Vec<T>`]: Where `T: PartialEq`.
pub trait Collection {
  /// The type of elements stored in the collection.
  type Item;

  /// Checks if an item is present in the collection.
  ///
  /// # Arguments
  ///
  /// * `point`: A reference to the item to check for containment.
  ///
  /// # Returns
  ///
  /// `true` if the item is found in the collection, `false` otherwise.
  fn contains(&self, point: &Self::Item) -> bool;

  /// Determines if the collection is empty.
  ///
  /// # Returns
  ///
  /// * `true` if the collection contains no items.
  /// * `false` if the collection contains one or more items.
  fn is_empty(&self) -> bool;
}

/// A trait for sets that support partial order relations, building upon [`Collection`].
///
/// This trait extends the basic [`Collection`] operations with methods specific to
/// partially ordered sets (posets). This includes checking the partial order
/// relation (`leq`), and computing various poset-specific subsets and elements.
///
/// # Type Parameters
///
/// * `Item`: The type of elements contained in the set, which are subject to the partial order.
pub trait Poset: Collection {
  /// Tests if one item is less than or equal to another according to the partial order.
  ///
  /// The `leq` relation (often denoted as $\le$) must satisfy reflexivity, antisymmetry,
  /// and transitivity.
  ///
  /// # Arguments
  ///
  /// * `a`: A reference to the first item.
  /// * `b`: A reference to the second item.
  ///
  /// # Returns
  ///
  /// * `Some(true)` if `a` is less than or equal to `b` ($a \le b$).
  /// * `Some(false)` if `a` is not less than or equal to `b` (and both are comparable in the set).
  /// * `None` if the relation cannot be determined (e.g., if one or both items are not considered
  ///   part of the poset for comparison, or if the specific poset implementation cannot compare
  ///   them).
  fn leq(&self, a: &Self::Item, b: &Self::Item) -> Option<bool>;

  /// Computes the upset of an item `a`.
  ///
  /// The upset of `a`, denoted $\uparrow a$, is the set of all items `x` in the poset
  /// such that $a \le x$.
  ///
  /// # Arguments
  ///
  /// * `a`: The item whose upset is to be computed.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all items `x` such that `a` is less than or equal to `x`.
  fn upset(&self, a: Self::Item) -> HashSet<Self::Item>;

  /// Computes the downset of an item `a`.
  ///
  /// The downset of `a`, denoted $\downarrow a$, is the set of all items `x` in the poset
  /// such that $x \le a$.
  ///
  /// # Arguments
  ///
  /// * `a`: The item whose downset is to be computed.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all items `x` such that `x` is less than or equal to `a`.
  fn downset(&self, a: Self::Item) -> HashSet<Self::Item>;

  /// Finds all minimal elements of the poset.
  ///
  /// An item `m` is minimal if there is no other item `x` in the poset such that $x < m$
  /// (i.e., $x \le m$ and $x \neq m$).
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all minimal elements of the poset.
  fn minimal_elements(&self) -> HashSet<Self::Item>;

  /// Finds all maximal elements of the poset.
  ///
  /// An item `m` is maximal if there is no other item `x` in the poset such that $m < x$
  /// (i.e., $m \le x$ and $m \neq x$).
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all maximal elements of the poset.
  fn maximal_elements(&self) -> HashSet<Self::Item>;

  /// Computes the join (least upper bound) of two items `a` and `b`, if it exists.
  ///
  /// The join $a \lor b$ is an item `j` such that $a \le j$ and $b \le j$, and for any
  /// other item `k` with $a \le k$ and $b \le k$, it holds that $j \le k$.
  ///
  /// # Arguments
  ///
  /// * `a`: The first item.
  /// * `b`: The second item.
  ///
  /// # Returns
  ///
  /// * `Some(join_element)` if the join of `a` and `b` exists.
  /// * `None` if the join does not exist or is not unique.
  fn join(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item>;

  /// Computes the meet (greatest lower bound) of two items `a` and `b`, if it exists.
  ///
  /// The meet $a \land b$ is an item `m` such that $m \le a$ and $m \le b$, and for any
  /// other item `k` with $k \le a$ and $k \le b$, it holds that $k \le m$.
  ///
  /// # Arguments
  ///
  /// * `a`: The first item.
  /// * `b`: The second item.
  ///
  /// # Returns
  ///
  /// * `Some(meet_element)` if the meet of `a` and `b` exists.
  /// * `None` if the meet does not exist or is not unique.
  fn meet(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item>;

  /// Finds all direct successors (covers) of an item `a`.
  ///
  /// An item `s` is a direct successor of `a` if $a < s$ (i.e., $a \le s$ and $a \neq s$)
  /// and there is no item `x` such that $a < x < s$.
  ///
  /// # Arguments
  ///
  /// * `a`: The item whose successors are to be found.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all direct successors of `a`.
  fn successors(&self, a: Self::Item) -> HashSet<Self::Item>;

  /// Finds all direct predecessors of an item `a`.
  ///
  /// An item `p` is a direct predecessor of `a` if $p < a$ (i.e., $p \le a$ and $p \neq a$)
  /// and there is no item `x` such that $p < x < a$.
  ///
  /// # Arguments
  ///
  /// * `a`: The item whose predecessors are to be found.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all direct predecessors of `a`.
  fn predecessors(&self, a: Self::Item) -> HashSet<Self::Item>;
}

impl<T: Hash + Eq + Clone, S: BuildHasher + Default> Collection for HashSet<T, S> {
  type Item = T;

  /// Checks if the `HashSet` contains the specified item.
  /// This is a direct wrapper around [`HashSet::contains`].
  fn contains(&self, point: &Self::Item) -> bool { Self::contains(self, point) }

  /// Checks if the `HashSet` is empty.
  /// This is a direct wrapper around [`HashSet::is_empty`].
  fn is_empty(&self) -> bool { Self::is_empty(self) }
}

impl<T: Ord + Clone> Collection for BTreeSet<T> {
  type Item = T;

  /// Checks if the `BTreeSet` contains the specified item.
  /// This is a direct wrapper around [`BTreeSet::contains`].
  fn contains(&self, point: &Self::Item) -> bool { Self::contains(self, point) }

  /// Checks if the `BTreeSet` is empty.
  /// This is a direct wrapper around [`BTreeSet::is_empty`].
  fn is_empty(&self) -> bool { Self::is_empty(self) }
}

impl<T: PartialEq> Collection for Vec<T> {
  type Item = T;

  /// Checks if the `Vec` contains the specified item by iterating through its elements.
  fn contains(&self, point: &Self::Item) -> bool { self.iter().any(|p| p == point) }

  /// Checks if the `Vec` is empty.
  /// This is a direct wrapper around [`Vec::is_empty`].
  fn is_empty(&self) -> bool { self.is_empty() }
}
