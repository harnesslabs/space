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

    // Convert indices to bit position using our helper function
    let bit_position = Self::blade_indices_to_bit(&indices);

    let mut value = Vector::<{ 2_usize.pow(N as u32) }, F>::zero();
    value.0[bit_position] = <F as Ring>::one();

    CliffordAlgebraElement { value, bilinear_space: Some(&self.bilinear_space) }
  }

  /// Maps a bit position to the corresponding basis blade indices.
  /// The bit position is interpreted as an index into the graded structure:
  /// - First C(n,0) positions are scalars
  /// - Next C(n,1) positions are vectors
  /// - Next C(n,2) positions are bivectors
  /// - And so on...
  fn bit_to_blade_indices(bits: usize) -> Vec<usize> {
    let mut remaining_bits = bits;
    let mut grade = 0;

    // Find which grade this element belongs to
    while grade <= N {
      let grade_size = binomial(N, grade);
      if remaining_bits < grade_size {
        break;
      }
      remaining_bits -= grade_size;
      grade += 1;
    }

    // Now we know the grade, we need to find which combination of indices
    // corresponds to the remaining_bits position within that grade
    let mut indices = Vec::with_capacity(grade);
    let mut current = 0;

    for _ in 0..grade {
      // Find the next index to include
      while current < N {
        let remaining_combinations = binomial(N - current - 1, grade - indices.len() - 1);
        if remaining_bits < remaining_combinations {
          indices.push(current);
          current += 1;
          break;
        }
        remaining_bits -= remaining_combinations;
        current += 1;
      }
    }

    indices
  }

  /// Maps a set of basis blade indices back to a bit position.
  /// This is the inverse of bit_to_blade_indices.
  fn blade_indices_to_bit(indices: &[usize]) -> usize {
    let grade = indices.len();
    let mut bit_position = 0;

    // Add up all the positions from lower grades
    for g in 0..grade {
      bit_position += binomial(N, g);
    }

    // Now add the position within this grade
    let mut remaining_bits = 0;
    for (i, &idx) in indices.iter().enumerate() {
      // For each index, add the number of combinations that come before it
      for j in if i == 0 { 0 } else { indices[i - 1] + 1 }..idx {
        remaining_bits += binomial(N - j - 1, grade - i - 1);
      }
    }

    bit_position + remaining_bits
  }
}

/// Computes the binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
  if k > n {
    return 0;
  }
  if k == 0 || k == n {
    return 1;
  }
  let k = std::cmp::min(k, n - k);
  let mut result = 1;
  for i in 1..=k {
    result = result * (n - k + i) / i;
  }
  result
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
      let indices = CliffordAlgebra::<F, N>::bit_to_blade_indices(idx);
      if indices.is_empty() {
        return Ok(());
      }

      write!(f, "e")?;
      for (i, &index) in indices.iter().enumerate() {
        // Convert number to Unicode subscript by converting each digit
        let num = index + 1;
        for digit in num.to_string().chars() {
          let subscript = match digit {
            '0' => "₀",
            '1' => "₁",
            '2' => "₂",
            '3' => "₃",
            '4' => "₄",
            '5' => "₅",
            '6' => "₆",
            '7' => "₇",
            '8' => "₈",
            '9' => "₉",
            _ => panic!("Invalid digit"),
          };
          write!(f, "{}", subscript)?;
        }

        if i < indices.len() - 1 {
          write!(f, "‚")?;
        }
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
macro_rules! impl_mul_scalar_clifford {
  ($($t:ty)*) => ($(
    impl<'a, const N: usize> Mul<CliffordAlgebraElement<'a, $t, N>> for $t
    where [(); 2_usize.pow(N as u32)]:
    {
      type Output = CliffordAlgebraElement<'a, $t, N>;

      fn mul(self, rhs: CliffordAlgebraElement<'a, $t, N>) -> Self::Output { rhs * self }
    }
  )*)
}

impl_mul_scalar_clifford!(f32);
impl_mul_scalar_clifford!(f64);

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

    let e12 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    println!("{e12}");

    let e13 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
    println!("{e13}");

    let e23 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    println!("{e23}");

    let e123 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    println!("{e123}");

    let sum = one + 2.0 * e1 + e2 * 3.0 + 4.0 * e3 + e12 * 5.0 + e13 * 6.0 + e23 * 7.0 + e123 * 8.0;
    println!("{sum}");
  }

  #[test]
  fn test_blade() {
    let algebra = clifford_algebra();

    let e1 = algebra.blade([1]);
    println!("{e1}");
  }

  #[test]
  fn test_blade_indices_to_bit() {
    // Test scalar (empty indices)
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[]), 0);

    // Test vectors
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[0]), 1);
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[1]), 2);
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[2]), 3);

    // Test bivectors
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[0, 1]), 4);
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[0, 2]), 5);
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[1, 2]), 6);

    // Test trivector
    assert_eq!(CliffordAlgebra::<f64, 3>::blade_indices_to_bit(&[0, 1, 2]), 7);
  }
}
