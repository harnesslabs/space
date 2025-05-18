//! # Trivial Module
//!
//! This module provides an implementation of a trivial module over an arbitrary ring.
//!
//! ## Mathematical Background
//!
//! In abstract algebra, a trivial module is a module that contains exactly one element:
//! the zero element. All operations on this element return the zero element itself:
//!
//! - Addition: $0 + 0 = 0$
//! - Negation: $-0 = 0$
//! - Scalar multiplication: $r \cdot 0 = 0$ for any ring element $r$
//!
//! ## Properties
//!
//! The trivial module satisfies all module axioms in the simplest possible way:
//!
//! - It forms an abelian group under addition (with only one element)
//! - Scalar multiplication is distributive over addition
//! - Scalar multiplication is compatible with ring multiplication
//!
//! ## Use Cases
//!
//! The trivial module serves several purposes:
//!
//! - As a base case in recursive constructions and mathematical proofs
//! - To represent the kernel or image of certain module homomorphisms
//! - For testing module-related algorithms with the simplest possible input
//! - As a terminal object in the category of R-modules
//!
//! ## Example
//!
//! ```
//! use harness_algebra::{modules::trivial::TrivialModule, prelude::*};
//!
//! // Create a trivial module over the integers
//! let m1: TrivialModule<i32> = TrivialModule::zero();
//! let m2: TrivialModule<i32> = TrivialModule::zero();
//!
//! // All operations return the same element
//! assert_eq!(m1 + m2, m1);
//! assert_eq!(m1 * 42, m1);
//! assert_eq!(-m1, m1);
//! ```

use super::*;

/// A trivial module over a ring.
///
/// This is a simple implementation of a module that has only one element.
/// It's useful as a base case or for testing purposes.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct TrivialModule<R> {
  pub(crate) _r: PhantomData<R>,
}

impl<R> Add for TrivialModule<R> {
  type Output = Self;

  fn add(self, _: Self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> AddAssign for TrivialModule<R> {
  fn add_assign(&mut self, _: Self) {}
}

impl<R> Sub for TrivialModule<R> {
  type Output = Self;

  fn sub(self, _: Self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> SubAssign for TrivialModule<R> {
  fn sub_assign(&mut self, _: Self) {}
}

impl<R> Neg for TrivialModule<R> {
  type Output = Self;

  fn neg(self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> Mul<R> for TrivialModule<R> {
  type Output = Self;

  fn mul(self, _: R) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> Zero for TrivialModule<R> {
  fn zero() -> Self { Self { _r: PhantomData } }

  fn is_zero(&self) -> bool { true }
}

impl<R: Ring> Additive for TrivialModule<R> {}

impl<R> Group for TrivialModule<R> {
  fn identity() -> Self { Self { _r: PhantomData } }

  fn inverse(&self) -> Self { Self { _r: PhantomData } }
}

impl<R: Ring> AbelianGroup for TrivialModule<R> {}
impl<R: Ring + Mul<Self>> LeftModule for TrivialModule<R> {
  type Ring = R;
}

impl<R: Ring + Mul<Self>> RightModule for TrivialModule<R> {
  type Ring = R;
}

impl<R: Ring + Mul<Self>> TwoSidedModule for TrivialModule<R> {
  type Ring = R;
}
