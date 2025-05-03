//! Semimodule theory abstractions and implementations.
//!
//! This module provides traits and implementations for semimodule theory concepts,
//! which generalize vector spaces by allowing the scalars to lie in a semiring rather than a field.
//!
//! # Key Concepts
//!
//! - **Semimodule**: A generalization of a vector space where the scalars are elements of a
//!   semiring instead of a field.
//! - **Left Semimodule**: A semimodule where scalar multiplication is defined from the left.
//! - **Right Semimodule**: A semimodule where scalar multiplication is defined from the right.
//! - **Two-Sided Semimodule**: A semimodule that is both a left and right semimodule over the same
//!   semiring.
//!
//! # Examples
//!
//! Common examples include:
//! - Vector spaces over semirings
//! - Matrices over semirings
//! - Polynomials over semirings
use crate::{arithmetic::Mul, ring::Semiring};

/// A trait representing a two-sided semimodule over a semiring.
///
/// A two-sided semimodule is a set that is both a left and right semimodule
/// over the same semiring. This trait does not enforce at the type level that
/// the `Semiring` associated types of the left and right semimodule implementations
/// are the same; it is up to the implementor to ensure this is the case.
///
/// # Note
///
/// Most commonly, the left and right actions coincide (for commutative semirings),
/// but this trait allows for the general, possibly noncommutative, case.
///
/// # See also
///
/// - [Semiring (Wikipedia)](https://en.wikipedia.org/wiki/Semiring)
/// - [Module (mathematics) - Generalizations](https://en.wikipedia.org/wiki/Module_(mathematics)#Generalizations)
pub trait TwoSidedSemimodule: LeftSemimodule + RightSemimodule {}

/// A trait representing a left semimodule over a semiring.
///
/// A left semimodule is a set that forms a commutative monoid under addition,
/// and has a left scalar multiplication operation that satisfies:
///
/// - Distributivity: s * (x + y) = s * x + s * y
/// - Compatibility: (s + t) * x = s * x + t * x
/// - Associativity: (s * t) * x = s * (t * x)
/// - Identity: 1 * x = x
/// - Zero: 0 * x = 0
///
/// where `s, t` are elements of the semiring, and `x, y` are elements of the semimodule.
///
/// # Associated Types
///
/// - `Semiring`: The type of the scalars, which must implement [`Semiring`].
///
/// # Trait Bounds
///
/// - `Self: Mul<Self::Semiring>`: Enables left scalar multiplication
pub trait LeftSemimodule
where Self: Mul<Self::Semiring> {
  /// The Semiring that this semimodule is defined over.
  type Semiring: Semiring;
}

/// A trait representing a right semimodule over a semiring.
///
/// A right semimodule is a set that forms a commutative monoid under addition,
/// and has a right scalar multiplication operation that satisfies:
///
/// - Distributivity: (x + y) * s = x * s + y * s
/// - Compatibility: x * (s + t) = x * s + x * t
/// - Associativity: x * (s * t) = (x * s) * t
/// - Identity: x * 1 = x
/// - Zero: x * 0 = 0
///
/// where `s, t` are elements of the semiring, and `x, y` are elements of the semimodule.
///
/// # Associated Types
///
/// - `Semiring`: The type of the scalars, which must implement [`Semiring`].
///
/// # Trait Bounds
///
/// - `Self: Mul<Self::Semiring>`: Enables right scalar multiplication
pub trait RightSemimodule
where Self: Mul<Self::Semiring> {
  /// The Semiring that this semimodule is defined over.
  type Semiring: Semiring;
}
