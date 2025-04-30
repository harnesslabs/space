//! Basic arithmetic traits and operations.
//!
//! This module provides fundamental arithmetic traits that are used throughout the algebra crate.
//! It re-exports standard arithmetic operations from [`std::ops`] and numeric traits from
//! [`num_traits`].
//!
//! # Examples
//!
//! ```
//! use algebra::arithmetic::{Additive, Multiplicative};
//!
//! // Types implementing Additive can be added and assigned
//! fn add<T: Additive>(a: T, b: T) -> T { a + b }
//!
//! // Types implementing Multiplicative can be multiplied and assigned
//! fn multiply<T: Multiplicative>(a: T, b: T) -> T { a * b }
//! ```

pub use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use num_traits::{One, Zero};

/// A trait for types that support addition (and comparison) operations.
///
/// This trait combines the basic requirements for types that can be added together:
/// - Addition operation with [`Add`] trait
/// - Addition assignment with [`AddAssign`] trait
/// - Equality comparison with [`PartialEq`] and [`Eq`]
///
/// # Examples
///
/// ```
/// use algebra::arithmetic::Additive;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyNumber(i32);
///
/// impl std::ops::Add for MyNumber {
///   type Output = Self;
///
///   fn add(self, rhs: Self) -> Self::Output { MyNumber(self.0 + rhs.0) }
/// }
///
/// impl std::ops::AddAssign for MyNumber {
///   fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
/// }
///
/// impl Additive for MyNumber {}
/// ```
pub trait Additive: Add<Output = Self> + AddAssign + PartialEq + Eq + Sized {}

/// A trait for types that support multiplication operations.
///
/// This trait combines the basic requirements for types that can be multiplied (and compared)
/// together:
/// - Multiplication operation with [`Mul`] trait
/// - Multiplication assignment with [`MulAssign`] trait
/// - Equality comparison with [`PartialEq`] and [`Eq`]
///
/// # Examples
///
/// ```
/// use algebra::arithmetic::Multiplicative;
///
/// #[derive(Copy, Clone, PartialEq, Eq)]
/// struct MyNumber(i32);
///
/// impl std::ops::Mul for MyNumber {
///   type Output = Self;
///
///   fn mul(self, rhs: Self) -> Self::Output { MyNumber(self.0 * rhs.0) }
/// }
///
/// impl std::ops::MulAssign for MyNumber {
///   fn mul_assign(&mut self, rhs: Self) { self.0 *= rhs.0; }
/// }
///
/// impl Multiplicative for MyNumber {}
/// ```
pub trait Multiplicative: Mul<Output = Self> + MulAssign + PartialEq + Eq + Sized {}
