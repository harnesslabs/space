//! Ring theory abstractions and implementations.
//!
//! This module provides traits and implementations for ring theory concepts,
//! including both general rings and fields (which are special types of rings).

use crate::{
  arithmetic::{Div, DivAssign, Multiplicative},
  group::AbelianGroup,
};

/// A trait representing a mathematical ring.
///
/// A ring is a set equipped with two binary operations (addition and multiplication)
/// satisfying properties analogous to those of addition and multiplication of integers.
/// This trait combines the requirements for an Abelian group with multiplicative properties.
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
pub trait Field: Ring + Div + DivAssign {
  /// Returns the multiplicative inverse of a non-zero element.
  ///
  /// # Panics
  ///
  /// This function may panic if called on the zero element.
  fn multiplicative_inverse(&self) -> Self;
}

#[macro_export]
macro_rules! impl_field {
  ($inner:ty) => {
    impl $crate::group::Group for $inner {
      fn identity() -> Self { 0.0 }

      fn inverse(&self) -> Self { -self }
    }

    impl $crate::group::AbelianGroup for $inner {}
    impl $crate::ring::Ring for $inner {
      fn one() -> Self { Self::one() }

      fn zero() -> Self { 0.0 }
    }
    impl $crate::ring::Field for $inner {
      fn multiplicative_inverse(&self) -> Self { self.recip() }
    }
  };
}

impl_field!(f32);
impl_field!(f64);
