//! Basic arithmetic traits and operations.
//!
//! This module provides fundamental arithmetic traits that are used throughout the algebra crate.
//! It re-exports standard arithmetic operations from [`std::ops`] and numeric traits from
//! [`num_traits`].
//!
//! # Examples
//!
//! ```
//! use cova_algebra::arithmetic::{Additive, Multiplicative};
//!
//! // Types implementing Additive can be added and assigned
//! fn add<T: Additive>(a: T, b: T) -> T { a + b }
//!
//! // Types implementing Multiplicative can be multiplied and assigned
//! fn multiply<T: Multiplicative>(a: T, b: T) -> T { a * b }
//! ```

use super::*;

pub mod modular;
pub mod primitive;

/// A trait for types that support addition (and comparison) operations.
///
/// This trait combines the basic requirements for types that can be added together:
/// - Addition operation with [`Add`] trait
/// - Addition assignment with [`AddAssign`] trait
/// - Equality comparison with [`PartialEq`]
///
///  # Examples
/// - All primitive numeric types implement this trait
/// - [`Boolean`](crate::algebras::boolean::Boolean) type implements this trait using bitwise
///   [`std::ops::BitXor`]
/// - Using the [`modular!`] macro, you can define a modular arithmetic type and it will implement
///   this trait.
/// - Using the [`prime_field!`] macro, you can define a prime field type and it will implement this
///   trait.
pub trait Additive:
  Add<Output = Self> + AddAssign + PartialEq + std::fmt::Debug + Sized + 'static {
}

/// A trait for types that support multiplication operations.
///
/// This trait combines the basic requirements for types that can be multiplied (and compared)
/// together:
/// - Multiplication operation with [`Mul`] trait
/// - Multiplication assignment with [`MulAssign`] trait
/// - Equality comparison with [`PartialEq`]
///
/// # Examples
/// - All primitive numeric types implement this trait
/// - [`Boolean`](crate::algebras::boolean::Boolean) type implements this trait using bitwise
///   [`std::ops::BitAnd`]
/// - Using the [`modular!`] macro, you can define a modular arithmetic type and it will implement
///   this trait.
/// - Using the [`prime_field!`] macro, you can define a prime field type and it will implement this
///   trait.
pub trait Multiplicative:
  Mul<Output = Self> + MulAssign + PartialEq + std::fmt::Debug + Sized + 'static {
}

/// Trait for types that have a concept of positive infinity.
pub trait Infinity {
  /// Returns the positive infinity value for the type.
  const INFINITY: Self;
}

pub trait ApproxZero {
  /// Returns true if the value is approximately zero.
  fn is_approx_zero(&self) -> bool;
}
