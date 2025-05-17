//! Vector space abstractions and implementations.
//!
//! This module provides traits and implementations for vector space concepts,
//! which are modules over fields. It includes a concrete implementation of
//! a fixed-size vector type.

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, Neg, Sub, SubAssign, Zero},
  group::{AbelianGroup, Group},
  module::{LeftModule, RightModule, TwoSidedModule},
  ring::Field,
};

/// A trait representing a vector space over a field.
///
/// A vector space is a module over a field, meaning it has both addition and
/// scalar multiplication operations, with the scalars coming from a field.
pub trait VectorSpace: TwoSidedModule
where <Self as TwoSidedModule>::Ring: Field {
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vector<const M: usize, F: Field>(pub [F; M]);

impl<const M: usize, F: Field + Copy> Default for Vector<M, F> {
  fn default() -> Self { Self([F::zero(); M]) }
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
  fn zero() -> Self { Self([F::zero(); M]) }

  fn is_zero(&self) -> bool { self.0.iter().all(|x| *x == F::zero()) }
}

impl<const M: usize, F: Field + Copy> AbelianGroup for Vector<M, F> {}

impl<const M: usize, F: Field + Copy + Mul<Self>> LeftModule for Vector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> RightModule for Vector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> TwoSidedModule for Vector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> VectorSpace for Vector<M, F> {}

impl<const M: usize, F: Field> From<[F; M]> for Vector<M, F> {
  fn from(components: [F; M]) -> Self { Self(components) }
}

impl<const M: usize, F: Field + Copy> From<&[F; M]> for Vector<M, F> {
  fn from(components: &[F; M]) -> Self { Self(*components) }
}

/// A dynamically-sized vector over a field `F`.
///
/// This structure represents a mathematical vector with components from a field `F`.
/// The dimension can be determined at runtime, making it flexible for various applications.
///
/// # Type Parameters
/// * `F` - A field type that implements the `Field` trait
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynVector<F: Field> {
  components: Vec<F>,
  dimension:  usize,
}

impl<F: Field> From<Vec<F>> for DynVector<F> {
  fn from(components: Vec<F>) -> Self {
    let dimension = components.len();
    Self { components, dimension }
  }
}

impl<const M: usize, F: Field + Copy> From<[F; M]> for DynVector<F> {
  fn from(components: [F; M]) -> Self {
    let dimension = M;
    Self { components: components.to_vec(), dimension }
  }
}

impl<F: Field + Clone> From<&[F]> for DynVector<F> {
  fn from(components: &[F]) -> Self {
    let dimension = components.len();
    Self { components: components.to_vec(), dimension }
  }
}

// TODO: This does handle the zero case but this is clunky as fuck and I hate it.
impl<F: Field + Copy> Add for DynVector<F> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    assert!((self.dimension == other.dimension) | (self.dimension == 0) | (other.dimension == 0));
    if self.dimension == 0 {
      return other;
    }
    if other.dimension == 0 {
      return self;
    }

    let mut sum = Self::zero();
    for i in 0..self.dimension {
      sum.components[i] = self.components[i] + other.components[i];
    }
    sum.dimension = self.dimension;
    sum
  }
}

impl<F: Field + Copy> AddAssign for DynVector<F> {
  fn add_assign(&mut self, rhs: Self) { *self = self.clone() + rhs }
}

impl<F: Field + Copy> Neg for DynVector<F> {
  type Output = Self;

  fn neg(self) -> Self::Output {
    let mut neg = Self::zero();
    for i in 0..self.dimension {
      neg.components[i] = -self.components[i];
    }
    neg
  }
}

impl<F: Field + Copy> Mul<F> for DynVector<F> {
  type Output = Self;

  fn mul(self, scalar: F) -> Self::Output {
    let mut scalar_multiple = Self::zero();
    for i in 0..self.dimension {
      scalar_multiple.components[i] = scalar * self.components[i];
    }
    scalar_multiple
  }
}

impl<F: Field + Copy> Sub for DynVector<F> {
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<F: Field + Copy> SubAssign for DynVector<F> {
  fn sub_assign(&mut self, rhs: Self) { *self = self.clone() - rhs }
}

impl<F: Field + Copy> Additive for DynVector<F> {}

impl<F: Field + Copy> Group for DynVector<F> {
  fn identity() -> Self { Self::zero() }

  fn inverse(&self) -> Self { -self.clone() }
}

// TODO: This is a bit odd
impl<F: Field + Copy> Zero for DynVector<F> {
  fn zero() -> Self {
    Self {
      components: {
        F::zero();
        vec![] as std::vec::Vec<F>
      },
      dimension:  0,
    }
  }

  fn is_zero(&self) -> bool { self.components.iter().all(|x| *x == F::zero()) }
}

impl<F: Field + Copy> AbelianGroup for DynVector<F> {}

impl<F: Field + Copy + Mul<Self>> LeftModule for DynVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> RightModule for DynVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> TwoSidedModule for DynVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> VectorSpace for DynVector<F> {}
