//! Basic arithmetic traits and operations.
//!
//! This module provides fundamental arithmetic traits that are used throughout the algebra crate.
//! It re-exports standard arithmetic operations from [`std::ops`] and numeric traits from
//! [`num_traits`].
//!
//! # Examples
//!
//! ```
//! use harness_algebra::arithmetic::{Additive, Multiplicative};
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
/// - Equality comparison with [`PartialEq`]
///
///  # Examples
/// - All primitive numeric types implement this trait
/// - [`Boolean`] type implements this trait using bitwise [`std::ops::BitXor`]
pub trait Additive: Add<Output = Self> + AddAssign + PartialEq + Sized {}

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
/// - [`Boolean`] type implements this trait using bitwise [`std::ops::BitAnd`]
pub trait Multiplicative: Mul<Output = Self> + MulAssign + PartialEq + Sized {}

/// A wrapper around `bool` that implements algebraic operations.
///
/// This type implements both [`Additive`] and [`Multiplicative`] traits using
/// bitwise operations:
/// - Addition is implemented as XOR (`^`)
/// - Multiplication is implemented as AND (`&`)
///
/// This makes `Boolean` a field with two elements, where:
/// - `false` is the additive identity (0)
/// - `true` is the multiplicative identity (1)
///
/// # Examples
///
/// ```
/// use harness_algebra::arithmetic::Boolean;
///
/// let a = Boolean(true);
/// let b = Boolean(false);
///
/// // Addition (XOR)
/// assert_eq!(a + b, Boolean(true));
/// assert_eq!(a + a, Boolean(false)); // a + a = 0
///
/// // Multiplication (AND)
/// assert_eq!(a * b, Boolean(false));
/// assert_eq!(a * a, Boolean(true)); // a * a = a
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Boolean(pub bool);

impl Add for Boolean {
  type Output = Self;

  /// Implements addition as XOR operation.
  ///
  /// This corresponds to the addition operation in the field GF(2).
  fn add(self, rhs: Self) -> Self::Output { Self(self.0 ^ rhs.0) }
}

impl AddAssign for Boolean {
  /// Implements addition assignment as XOR operation.
  fn add_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
}

impl Mul for Boolean {
  type Output = Self;

  /// Implements multiplication as AND operation.
  ///
  /// This corresponds to the multiplication operation in the field GF(2).
  fn mul(self, rhs: Self) -> Self::Output { Self(self.0 && rhs.0) }
}

impl MulAssign for Boolean {
  /// Implements multiplication assignment as AND operation.
  fn mul_assign(&mut self, rhs: Self) { self.0 &= rhs.0; }
}

impl Additive for Boolean {}
impl Multiplicative for Boolean {}

// Implement Additive for all primitive numeric types
impl Additive for u8 {}
impl Additive for u16 {}
impl Additive for u32 {}
impl Additive for u64 {}
impl Additive for u128 {}
impl Additive for usize {}

impl Additive for i8 {}
impl Additive for i16 {}
impl Additive for i32 {}
impl Additive for i64 {}
impl Additive for i128 {}
impl Additive for isize {}

impl Additive for f32 {}
impl Additive for f64 {}

// Implement Multiplicative for all primitive numeric types
impl Multiplicative for u8 {}
impl Multiplicative for u16 {}
impl Multiplicative for u32 {}
impl Multiplicative for u64 {}
impl Multiplicative for u128 {}
impl Multiplicative for usize {}

impl Multiplicative for i8 {}
impl Multiplicative for i16 {}
impl Multiplicative for i32 {}
impl Multiplicative for i64 {}
impl Multiplicative for i128 {}
impl Multiplicative for isize {}

impl Multiplicative for f32 {}
impl Multiplicative for f64 {}
