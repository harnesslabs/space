//! Module theory abstractions and implementations.
//!
//! This module provides traits and implementations for module theory concepts,
//! which generalize vector spaces by allowing the scalars to lie in a ring rather than a field.

use core::marker::PhantomData;

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, Neg, Sub, SubAssign, Zero},
  group::{AbelianGroup, Group},
  ring::Ring,
};

/// A trait representing a two-sided module over a ring.
///
/// A two-sided module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on both the left and right.
pub trait TwoSidedModule: LeftModule + RightModule {
  /// The ring over which this module is defined.
  type Ring: Ring;
}

/// A trait representing a left module over a ring.
///
/// A left module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on the left.
pub trait LeftModule: AbelianGroup
where Self::Ring: Mul<Self> {
  /// The ring over which this module is defined.
  type Ring: Ring;
}

/// A trait representing a right module over a ring.
///
/// A right module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on the right.
pub trait RightModule: AbelianGroup
where Self::Ring: Mul<Self> {
  /// The ring over which this module is defined.
  type Ring: Ring;
}
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
