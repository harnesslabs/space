use std::{
  fmt::Debug,
  ops::{Add, AddAssign, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::Zero;

use super::Algebra;
use crate::{
  arithmetic::{Additive, Mul, Multiplicative},
  group::{AbelianGroup, Group},
  module::Module,
  ring::{Field, Ring},
  vector::{Vector, VectorSpace},
};

// TODO: We are assuming this is in the diagonal basis.
#[derive(Debug, PartialEq, Eq)]
pub struct BilinearSpace<F: Field, const N: usize> {
  vector_basis: Vector<N, F>,
  coefficients: Vector<N, F>,
}

impl<F: Field + Copy, const N: usize> BilinearSpace<F, N> {
  pub const fn new(vector_basis: Vector<N, F>, coefficients: Vector<N, F>) -> Self {
    Self { vector_basis, coefficients }
  }

  pub fn new_standard_basis(vector_basis: Vector<N, F>) -> Self {
    let mut coefficients = Vector::<N, F>::zero();
    for i in 0..N {
      coefficients.0[i] = <F as Ring>::one();
    }
    Self::new(vector_basis, coefficients)
  }

  pub fn evaluate(&self, v: &Vector<N, F>, w: &Vector<N, F>) -> F {
    let mut result = <F as Ring>::zero();
    for i in 0..N {
      result = result + self.coefficients.0[i] * v.0[i] * w.0[i];
    }
    result
  }
}

pub struct CliffordAlgebra<F: Field, const N: usize> {
  bilinear_space: BilinearSpace<F, N>,
}

impl<F: Field + Copy, const N: usize> CliffordAlgebra<F, N>
where [(); 2_usize.pow(N as u32)]:
{
  pub const fn new(bilinear_space: BilinearSpace<F, N>) -> Self { Self { bilinear_space } }

  pub fn element(
    &self,
    value: Vector<{ 2_usize.pow(N as u32) }, F>,
  ) -> CliffordAlgebraElement<'_, F, N> {
    CliffordAlgebraElement { value, bilinear_space: Some(&self.bilinear_space) }
  }

  pub fn basic_element<const I: usize>(
    &self,
    indices: [usize; I],
  ) -> CliffordAlgebraElement<'_, F, N> {
    let mut value = Vector::<{ 2_usize.pow(N as u32) }, F>::zero();
    todo!()
  }
}

// TODO: We can make both an option and make the case where both are none the zero element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CliffordAlgebraElement<'a, F: Field, const N: usize>
where [(); 2_usize.pow(N as u32)]: {
  value:          Vector<{ 2_usize.pow(N as u32) }, F>,
  bilinear_space: Option<&'a BilinearSpace<F, N>>,
}

// TODO: All of these impls should check the same bilinear space. This should probably be a compile
// time thing.

impl<'a, F: Field + Copy + Debug, const N: usize> Add for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    assert_eq!(self.bilinear_space, other.bilinear_space);
    Self { value: self.value + other.value, bilinear_space: self.bilinear_space }
  }
}

impl<'a, F: Field + Copy + Debug, const N: usize> AddAssign for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<'a, F: Field + Copy, const N: usize> Neg for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Output = Self;

  fn neg(self) -> Self::Output { Self { value: -self.value, bilinear_space: self.bilinear_space } }
}

impl<'a, F: Field + Copy + Debug, const N: usize> Sub for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<'a, F: Field + Copy + Debug, const N: usize> SubAssign for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl<'a, F: Field + Copy, const N: usize> Mul for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    let value = todo!();

    Self { value, bilinear_space: self.bilinear_space }
  }
}

impl<'a, F: Field + Copy, const N: usize> MulAssign for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<'a, F: Field + Copy, const N: usize> Mul<F> for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Output = Self;

  fn mul(self, rhs: F) -> Self::Output {
    Self { value: self.value * rhs, bilinear_space: self.bilinear_space }
  }
}

impl<'a, F: Field + Copy, const N: usize> Multiplicative for CliffordAlgebraElement<'a, F, N> where [(); 2_usize.pow(N as u32)]: {}

// TODO: This is weird... i'll use option to note the zero element.
impl<'a, F: Field + Copy + Debug, const N: usize> Zero for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn zero() -> Self {
    Self { value: Vector::<{ 2_usize.pow(N as u32) }, F>::zero(), bilinear_space: None }
  }

  fn is_zero(&self) -> bool { self.value.is_zero() }
}

impl<'a, F: Field + Copy + Debug, const N: usize> Additive for CliffordAlgebraElement<'a, F, N> where [(); 2_usize.pow(N as u32)]: {}
impl<'a, F: Field + Copy + Debug, const N: usize> Group for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn identity() -> Self {
    CliffordAlgebraElement {
      value:          Vector::<{ 2_usize.pow(N as u32) }, F>::zero(),
      bilinear_space: None,
    }
  }

  fn inverse(&self) -> Self { Self { value: -self.value, bilinear_space: self.bilinear_space } }
}
impl<'a, F: Field + Copy + Debug, const N: usize> AbelianGroup for CliffordAlgebraElement<'a, F, N> where [(); 2_usize.pow(N as u32)]: {}
impl<'a, F: Field + Copy + Debug, const N: usize> Module for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Ring = F;
}
impl<'a, F: Field + Copy + Debug, const N: usize> VectorSpace for CliffordAlgebraElement<'a, F, N> where [(); 2_usize.pow(N as u32)]: {}
impl<'a, F: Field + Copy + Debug, const N: usize> Algebra for CliffordAlgebraElement<'a, F, N> where [(); 2_usize.pow(N as u32)]: {}

#[cfg(test)]
mod tests {
  use super::*;

  fn clifford_algebra() -> CliffordAlgebra<f64, 3> {
    let bilinear_space = BilinearSpace::new_standard_basis(Vector::<3, f64>([1.0, 1.0, -1.0]));
    CliffordAlgebra::new(bilinear_space)
  }

  #[test]
  fn test_clifford_algebra() { let algebra = clifford_algebra(); }
}
