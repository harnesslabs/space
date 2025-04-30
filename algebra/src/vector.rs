//! Vector space abstractions and implementations.
//!
//! This module provides traits and implementations for vector space concepts,
//! which are modules over fields. It includes a concrete implementation of
//! a fixed-size vector type.

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, Neg, Sub, SubAssign, Zero},
  group::{AbelianGroup, Group},
  module::Module,
  ring::{Field, Ring},
};

/// A trait representing a vector space over a field.
///
/// A vector space is a module over a field, meaning it has both addition and
/// scalar multiplication operations, with the scalars coming from a field.
pub trait VectorSpace: Module
where Self::Ring: Field {
}

/// A fixed-size vector over a field.
///
/// This is a concrete implementation of a vector space, where vectors have
/// a fixed number of components and the scalars come from a field.
///
/// ```
/// use harness_algebra::{ring::Field, vector::Vector};
///
/// let v = Vector::<3, f64>([1.0, 2.0, 3.0]);
/// let w = Vector::<3, f64>([4.0, 5.0, 6.0]);
/// let sum = v + w;
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<const M: usize, F: Field>(pub [F; M]);

impl<const M: usize, F: Field + Copy> Default for Vector<M, F> {
  fn default() -> Self { Self([<F as Ring>::zero(); M]) }
}

impl<const M: usize, F: Field + Copy> Add for Vector<M, F> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    let mut sum = Self::zero();
    for i in 0..M {
      sum.0[i] = self.0[i] + other.0[i];
    }
    sum
  }
}

impl<const M: usize, F: Field + Copy> AddAssign for Vector<M, F> {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs }
}

impl<const M: usize, F: Field + Copy> Neg for Vector<M, F> {
  type Output = Self;

  fn neg(self) -> Self::Output {
    let mut neg = Self::zero();
    for i in 0..M {
      neg.0[i] = -self.0[i];
    }
    neg
  }
}

impl<const M: usize, F: Field + Copy> Mul<F> for Vector<M, F> {
  type Output = Self;

  fn mul(self, scalar: F) -> Self::Output {
    let mut scalar_multiple = Self::zero();
    for i in 0..M {
      scalar_multiple.0[i] = scalar * self.0[i];
    }
    scalar_multiple
  }
}

impl<const M: usize, F: Field + Copy> Sub for Vector<M, F> {
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<const M: usize, F: Field + Copy> SubAssign for Vector<M, F> {
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs }
}

impl<const M: usize, F: Field + Copy> Additive for Vector<M, F> {}

impl<const M: usize, F: Field + Copy> Group for Vector<M, F> {
  fn identity() -> Self { Self::zero() }

  fn inverse(&self) -> Self { -*self }
}

impl<const M: usize, F: Field + Copy> Zero for Vector<M, F> {
  fn zero() -> Self { Self([<F as Ring>::zero(); M]) }

  fn is_zero(&self) -> bool { self.0.iter().all(|x| *x == <F as Ring>::zero()) }
}

impl<const M: usize, F: Field + Copy> AbelianGroup for Vector<M, F> {}

impl<const M: usize, F: Field + Copy> Module for Vector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy> VectorSpace for Vector<M, F> {}
