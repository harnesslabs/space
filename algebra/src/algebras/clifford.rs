use std::{
  fmt::{Debug, Display, Formatter},
  ops::{Add, AddAssign, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::Zero;

use super::Algebra;
use crate::{
  arithmetic::{Additive, Mul, Multiplicative},
  group::{AbelianGroup, Group},
  module::{LeftModule, RightModule, TwoSidedModule},
  ring::{Field, Ring},
  vector::{Vector, VectorSpace},
};

// TODO: We are assuming this is in the diagonal basis.
#[derive(Debug, PartialEq, Eq)]
pub struct BilinearSpace<F: Field, const N: usize> {
  coefficients: Vector<N, F>,
}

impl<F: Field + Copy, const N: usize> BilinearSpace<F, N> {
  pub const fn new(coefficients: Vector<N, F>) -> Self { Self { coefficients } }

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

  pub fn blade<const I: usize>(&self, indices: [usize; I]) -> CliffordAlgebraElement<'_, F, N> {
    // Validate indices are in range and sorted
    for i in 1..I {
      assert!(indices[i - 1] < indices[i] && indices[i] < N, "Indices must be sorted and in range");
    }

    // The position in the 2^N array is determined by the binary representation
    // where 1s indicate which basis vectors are included
    let mut pos = 0usize;
    for &idx in indices.iter() {
      pos |= 1 << idx;
    }

    let mut value = Vector::<{ 2_usize.pow(N as u32) }, F>::zero();
    value.0[pos] = <F as Ring>::one();

    CliffordAlgebraElement { value, bilinear_space: Some(&self.bilinear_space) }
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
impl<'a, F: Field + Copy + Debug + Mul<CliffordAlgebraElement<'a, F, N>>, const N: usize> LeftModule
  for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Ring = F;
}
impl<'a, F: Field + Copy + Debug + Mul<CliffordAlgebraElement<'a, F, N>>, const N: usize>
  RightModule for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Ring = F;
}
impl<'a, F: Field + Copy + Debug + Mul<CliffordAlgebraElement<'a, F, N>>, const N: usize>
  TwoSidedModule for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  type Ring = F;
}
impl<'a, F: Field + Copy + Debug + Mul<CliffordAlgebraElement<'a, F, N>>, const N: usize>
  VectorSpace for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
}
impl<'a, F: Field + Copy + Debug + Mul<CliffordAlgebraElement<'a, F, N>>, const N: usize> Algebra
  for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
}

impl<'a, F: Field + Copy + Display, const N: usize> Display for CliffordAlgebraElement<'a, F, N>
where [(); 2_usize.pow(N as u32)]:
{
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    let mut first = true;

    // Helper function to write basis element
    let write_basis = |f: &mut Formatter<'_>, idx: usize| -> std::fmt::Result {
      let mut bits = idx;
      let mut first_e = true;

      for i in 0..N {
        if bits & 1 == 1 {
          if !first_e {
            write!(f, " ")?;
          }
          write!(f, "e_{}", i + 1)?;
          first_e = false;
        }
        bits >>= 1;
      }
      Ok(())
    };

    // Print each grade in order
    for grade in 0..=N {
      // For each possible combination of 'grade' basis vectors
      for idx in 0..2usize.pow(N as u32) {
        if idx.count_ones() as usize == grade && !self.value.0[idx].is_zero() {
          if !first {
            write!(f, " + ")?;
          }
          write!(f, "{}", self.value.0[idx])?;
          if grade > 0 {
            write!(f, " ")?;
            write_basis(f, idx)?;
          }
          first = false;
        }
      }
    }

    if first {
      write!(f, "0")?;
    }
    Ok(())
  }
}

#[macro_export]
macro_rules! impl_mul_scalar {
  ($($t:ty)*) => ($(
    impl<'a, const N: usize> Mul<CliffordAlgebraElement<'a, $t, N>> for $t
    where [(); 2_usize.pow(N as u32)]:
    {
      type Output = CliffordAlgebraElement<'a, $t, N>;

      fn mul(self, rhs: CliffordAlgebraElement<'a, $t, N>) -> Self::Output { rhs * self }
    }
  )*)
}

crate::impl_mul_scalar_generic!(f32, CliffordAlgebraElement<'a, f32, N>);

// impl_mul_scalar!(f32);
impl_mul_scalar!(f64);

#[cfg(test)]
mod tests {
  use super::*;

  fn clifford_algebra() -> CliffordAlgebra<f64, 3> {
    let bilinear_space = BilinearSpace::new(Vector::<3, f64>([1.0, 1.0, -1.0]));
    CliffordAlgebra::new(bilinear_space)
  }

  #[test]
  fn test_operations() {
    let algebra = clifford_algebra();

    let one = algebra.element(Vector::<8, f64>([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    println!("{one}");

    let e1 = algebra.element(Vector::<8, f64>([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    println!("{e1}");

    let e2 = algebra.element(Vector::<8, f64>([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    println!("{e2}");

    let e3 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
    println!("{e3}");

    let sum = one + 2.0 * e1 + e2 + e3;
    println!("{sum}");

    // let e1 = algebra.blade([1]);
    // dbg!(&e1);
    // let e2 = algebra.blade([2]);
    // dbg!(&e2);
    // let e3 = algebra.blade([3]);
    // dbg!(&e3);

    // let e12 = e1 * e2;
    // let e123 = e12 * e3;
  }
}
