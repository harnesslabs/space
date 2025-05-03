//! Semiring theory abstractions and implementations.
//!
//! This module provides traits and implementations for semiring theory concepts.
//! Semirings are generalisations of rings that omit the additive inverse property.
//! This means that semirings do not necessarily have "negative elements".
//!
//! # Examples
//!
//! Common examples of semirings include:
//! - Natural numbers (ℕ, +, ×)
//! - Boolean algebra ({0,1}, ∨, ∧)
//! - Tropical semiring (ℝ ∪ {∞}, min, +)
//! - Probability semiring (ℝ₊, +, ×)

use crate::arithmetic::{Additive, Multiplicative, One, Zero};

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
/// If you want a structure with an additive inverse, use the [`ring::Ring`] trait instead, since it
/// has the abelian group trait bound. If you only need addition to be associative and commutative
/// (but without an additive identity), use the [`semigroup::Semigroup`] trait instead.
pub trait Semiring: Additive + Multiplicative + Zero + One {
  /// Returns the additive identity element of the semiring.
  fn zero() -> Self;

  /// Returns the multiplicative identity element of the semiring.
  fn one() -> Self;
}
