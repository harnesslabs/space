//! Ring theory abstractions and implementations.
//!
//! This module provides traits and implementations for ring theory concepts,
//! including both general rings and fields (which are special types of rings).
//!
//! # Examples
//!
//! ```
//! use algebra::ring::{Field, Ring};
//!
//! #[derive(Copy, Clone, PartialEq, Eq)]
//! struct MyRing(i32);
//!
//! impl Ring for MyRing {
//!   fn one() -> Self { MyRing(1) }
//!
//!   fn zero() -> Self { MyRing(0) }
//! }
//!
//! impl Field for MyRing {
//!   fn multiplicative_inverse(&self) -> Self { MyRing(1 / self.0) }
//! }
//! ```

use crate::{
  arithmetic::{Div, DivAssign, Multiplicative},
  group::AbelianGroup,
};

/// A trait representing a mathematical ring.
///
/// A ring is a set equipped with two binary operations (addition and multiplication)
/// satisfying properties analogous to those of addition and multiplication of integers.
/// This trait combines the requirements for an Abelian group with multiplicative properties.
///
/// # Examples
///
/// ```
/// use algebra::ring::Ring;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyRing(i32);
///
/// impl Ring for MyRing {
///   fn one() -> Self { MyRing(1) }
///
///   fn zero() -> Self { MyRing(0) }
/// }
/// ```
pub trait Ring: AbelianGroup + Multiplicative {
  /// Returns the multiplicative identity element of the ring.
  fn one() -> Self;

  /// Returns the additive identity element of the ring.
  fn zero() -> Self;
}

/// A trait representing a mathematical field.
///
/// A field is a set on which addition, subtraction, multiplication, and division
/// are defined and behave as the corresponding operations on rational and real numbers.
/// Every non-zero element has a multiplicative inverse.
///
/// # Examples
///
/// ```
/// use algebra::ring::Field;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyField(f64);
///
/// impl Field for MyField {
///   fn multiplicative_inverse(&self) -> Self { MyField(1.0 / self.0) }
/// }
/// ```
pub trait Field: Ring + Div + DivAssign {
  /// Returns the multiplicative inverse of a non-zero element.
  ///
  /// # Panics
  ///
  /// This function may panic if called on the zero element.
  fn multiplicative_inverse(&self) -> Self;
}
