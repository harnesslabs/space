//! Group theory abstractions and implementations.
//!
//! This module provides traits and implementations for group theory concepts,
//! including both Abelian (commutative) and non-Abelian groups.
//!
//! # Examples
//!
//! ```
//! use algebra::group::{AbelianGroup, Group};
//!
//! #[derive(Copy, Clone, PartialEq, Eq)]
//! struct MyGroup(i32);
//!
//! impl Group for MyGroup {
//!   fn identity() -> Self { MyGroup(0) }
//!
//!   fn inverse(&self) -> Self { MyGroup(-self.0) }
//! }
//!
//! impl AbelianGroup for MyGroup {}
//! ```

use crate::arithmetic::{Additive, Div, DivAssign, Multiplicative, Neg, One, Sub, SubAssign, Zero};

/// A trait representing a mathematical group.
///
/// A group is a set equipped with an operation that combines any two of its elements
/// to form a third element, satisfying four conditions called the group axioms:
/// closure, associativity, identity, and invertibility.
///
/// # Examples
///
/// ```
/// use algebra::group::Group;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyGroup(i32);
///
/// impl Group for MyGroup {
///   fn identity() -> Self { MyGroup(0) }
///
///   fn inverse(&self) -> Self { MyGroup(-self.0) }
/// }
/// ```
pub trait Group {
  /// Returns the identity element of the group.
  fn identity() -> Self;

  /// Returns the inverse of an element.
  fn inverse(&self) -> Self;
}

/// A trait representing an Abelian (commutative) group.
///
/// An Abelian group is a group where the group operation is commutative.
/// This trait combines the requirements for a group with additional operations
/// that are natural for commutative groups. We mark this as an [`Additive`] structure since this is
/// typical notation for Abelian groups.
///
/// # Examples
///
/// ```
/// use algebra::group::AbelianGroup;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyAbelianGroup(i32);
///
/// // Implement required traits...
/// impl AbelianGroup for MyAbelianGroup {}
/// ```
pub trait AbelianGroup:
  Group + Zero + Additive + Neg<Output = Self> + Sub<Output = Self> + SubAssign {
}

/// A trait representing a non-Abelian group.
///
/// A non-Abelian group is a group where the group operation is not necessarily commutative.
/// This trait combines the requirements for a group with additional operations
/// that are natural for non-commutative groups. We mark this as a [`Multiplicative`] structure
/// since this is typical notation for non-Abelian groups. However, it should be noted that a
/// [`NonAbelianGroup`] group cannot be an [`AbelianGroup`]
///
/// # Examples
///
/// ```
/// use algebra::group::NonAbelianGroup;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyNonAbelianGroup(i32);
///
/// // Implement required traits...
/// impl NonAbelianGroup for MyNonAbelianGroup {}
/// ```
pub trait NonAbelianGroup: Group + One + Multiplicative + Div<Output = Self> + DivAssign {}
