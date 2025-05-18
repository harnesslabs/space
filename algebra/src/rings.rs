//! Ring theory abstractions and implementations.
//!
//! This module provides traits and implementations for ring theory concepts,
//! including both general rings and fields (which are special types of rings).

use super::*;
use crate::groups::AbelianGroup;

/// A trait representing a mathematical ring.
///
/// A ring is a set equipped with two binary operations (addition and multiplication)
/// satisfying properties analogous to those of addition and multiplication of integers.
/// This trait combines the requirements for an Abelian group with multiplicative properties.
pub trait Ring: AbelianGroup + Multiplicative + One {}

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

macro_rules! impl_field {
  ($inner:ty) => {
    impl $crate::groups::Group for $inner {
      fn identity() -> Self { 0.0 }

      fn inverse(&self) -> Self { -self }
    }

    impl $crate::groups::AbelianGroup for $inner {}
    impl $crate::rings::Ring for $inner {}
    impl $crate::rings::Field for $inner {
      fn multiplicative_inverse(&self) -> Self { self.recip() }
    }
  };
}

impl_field!(f32);
impl_field!(f64);

/// A trait representing a mathematical semiring.
///
/// A semiring is a set equipped with two binary operations (addition and multiplication)
/// satisfying properties of distributivity and associativity analogous to those of addition and
/// multiplication of integers. This trait combines the requirements for an Abelian monoid with
/// multiplicative properties.
///
/// # Requirements
///
/// A semiring (R, +, ·) must satisfy:
/// 1. (R, +) is a commutative monoid with identity element 0
/// 2. (R, ·) is a monoid with identity element 1
/// 3. Multiplication distributes over addition:
///    - Left distributivity: a·(b + c) = a·b + a·c
///    - Right distributivity: (a + b)·c = a·c + b·c
/// 4. Multiplication by 0 annihilates R: 0·a = a·0 = 0
///
/// # Implementation Notes
///
/// The distributive properties are enforced by the combination of the `Additive` and
/// `Multiplicative` traits. Implementors must ensure that their implementations satisfy these
/// properties. Semirings are not groups because they do not have additive inverses.
///
/// If you want a structure with an additive inverse, use the Ring trait instead, since it
/// has the abelian group trait bound. If you only need addition to be associative and commutative
/// (but without an additive identity), use the semiring trait.
///
/// # Examples
///
/// Common examples of semirings include:
/// - Natural numbers (ℕ, +, ×)
/// - Tropical semiring (ℝ ∪ {∞}, min, +)
/// - Probability semiring (ℝ₊, +, ×)
pub trait Semiring: Additive + Multiplicative + Zero + One {}
