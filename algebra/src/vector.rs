use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, Neg, Sub, SubAssign, Zero},
  group::{AbelianGroup, Group},
  module::Module,
  ring::{Field, Ring},
};

pub trait VectorSpace: Module
where Self::Ring: Field {
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<const M: usize, F: Field>(pub [F; M]);

impl<const M: usize, F: Field> Default for Vector<M, F> {
  fn default() -> Self { Self([<F as Ring>::zero(); M]) }
}

impl<const M: usize, F: Field> Add for Vector<M, F> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    let mut sum = Self::default();
    for i in 0..M {
      sum.0[i] = self.0[i] + other.0[i];
    }
    sum
  }
}

impl<const M: usize, F: Field> AddAssign for Vector<M, F> {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs }
}

impl<const M: usize, F: Field> Neg for Vector<M, F> {
  type Output = Self;

  fn neg(self) -> Self::Output {
    let mut neg = Self::default();
    for i in 0..M {
      neg.0[i] = -self.0[i];
    }
    neg
  }
}

impl<const M: usize, F: Field> Mul<F> for Vector<M, F> {
  type Output = Self;

  fn mul(self, scalar: F) -> Self::Output {
    let mut scalar_multiple = Self::default();
    for i in 0..M {
      scalar_multiple.0[i] = scalar * self.0[i];
    }
    scalar_multiple
  }
}

impl<const M: usize, F: Field> Sub for Vector<M, F> {
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<const M: usize, F: Field> SubAssign for Vector<M, F> {
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs }
}

impl<const M: usize, F: Field> Additive for Vector<M, F> {}

impl<const M: usize, F: Field> Group for Vector<M, F> {
  fn identity() -> Self { Self::default() }

  fn inverse(&self) -> Self { -*self }
}

impl<const M: usize, F: Field> Zero for Vector<M, F> {
  fn zero() -> Self { Self([<F as Ring>::zero(); M]) }

  fn is_zero(&self) -> bool { self.0.iter().all(|x| *x == <F as Ring>::zero()) }
}

impl<const M: usize, F: Field> AbelianGroup for Vector<M, F> {}

impl<const M: usize, F: Field> Module for Vector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field> VectorSpace for Vector<M, F> {}
