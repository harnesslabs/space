use std::fmt::{Debug, Display, Formatter};

use crate::{
  algebras::Algebra,
  arithmetic::{
    Add, AddAssign, Additive, Mul, MulAssign, Multiplicative, Neg, Sub, SubAssign, Zero,
  },
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
      result += self.coefficients.0[i] * v.0[i] * w.0[i];
    }
    result
  }
}

pub struct CliffordAlgebra<F: Field, const N: usize> {
  bilinear_space: BilinearSpace<F, N>,
}

impl<F: Field + Copy, const N: usize> CliffordAlgebra<F, N>
where [(); 1 << N]:
{
  pub const fn new(bilinear_space: BilinearSpace<F, N>) -> Self { Self { bilinear_space } }

  pub const fn element(&self, value: Vector<{ 1 << N }, F>) -> CliffordAlgebraElement<'_, F, N> {
    CliffordAlgebraElement { value, bilinear_space: Some(&self.bilinear_space) }
  }

  pub fn blade<const I: usize>(&self, indices: [usize; I]) -> CliffordAlgebraElement<'_, F, N> {
    // Validate indices are in range and sorted
    for i in 1..I {
      assert!(
        indices[i - 1] < indices[i] && indices[i] < N,
        "Indices must be sorted and in
    range"
      );
    }

    // Convert indices to bit position using our helper function
    let bit_position = Self::blade_indices_to_bit(&indices);

    let mut value = Vector::<{ 1 << N }, F>::zero();
    value.0[bit_position] = F::one();

    CliffordAlgebraElement { value, bilinear_space: Some(&self.bilinear_space) }
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
where [(); 1 << N]: {
  value:          Vector<{ 1 << N }, F>,
  bilinear_space: Option<&'a BilinearSpace<F, N>>,
}

// TODO: All of these impls should check the same bilinear space. This should probably be a compile
// time thing.

impl<F: Field + Copy + Debug, const N: usize> Add for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    assert_eq!(self.bilinear_space, other.bilinear_space);
    Self { value: self.value + other.value, bilinear_space: self.bilinear_space }
  }
}

impl<F: Field + Copy + Debug, const N: usize> AddAssign for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<F: Field + Copy, const N: usize> Neg for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn neg(self) -> Self::Output { Self { value: -self.value, bilinear_space: self.bilinear_space } }
}

impl<F: Field + Copy + Debug, const N: usize> Sub for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<F: Field + Copy + Debug, const N: usize> SubAssign for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl<F: Field + Copy + From<i32> + Debug, const N: usize> Mul for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    assert_eq!(self.bilinear_space, other.bilinear_space);
    let bilinear_space = self.bilinear_space.expect("Both elements must have a bilinear space");

    let mut result = Vector::<{ 1 << N }, F>::zero();

    // For each non-zero component in the first element
    for i in 0..(1 << N) {
      if self.value.0[i].is_zero() {
        continue;
      }

      // For each non-zero component in the second element
      for j in 0..(1 << N) {
        if other.value.0[j].is_zero() {
          continue;
        }

        // Get the indices for both basis blades
        let left_indices = Self::bit_to_blade_indices(i);
        let right_indices = Self::bit_to_blade_indices(j);

        // Calculate the sign and product indices
        let (sign, product_indices) =
          multiply_blades(&left_indices, &right_indices, bilinear_space);

        // Calculate the coefficient
        let coefficient = if sign {
          self.value.0[i] * other.value.0[j]
        } else {
          (self.value.0[i] * other.value.0[j]).inverse()
        };

        // Add to the result
        let product_bit = Self::blade_indices_to_bit(&product_indices);
        result.0[product_bit] += coefficient;
      }
    }

    Self { value: result, bilinear_space: self.bilinear_space }
  }
}

impl<F: Field + Copy, const N: usize> MulAssign for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<F: Field + Copy, const N: usize> Mul<F> for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn mul(self, rhs: F) -> Self::Output {
    Self { value: self.value * rhs, bilinear_space: self.bilinear_space }
  }
}

impl<F: Field + Copy + Debug, const N: usize> Multiplicative for CliffordAlgebraElement<'_, F, N> where [(); 1 << N]: {}

// TODO: This is weird... i'll use option to note the zero element.
impl<F: Field + Copy + Debug, const N: usize> Zero for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn zero() -> Self { Self { value: Vector::<{ 1 << N }, F>::zero(), bilinear_space: None } }

  fn is_zero(&self) -> bool { self.value.is_zero() }
}

impl<F: Field + Copy + Debug, const N: usize> Additive for CliffordAlgebraElement<'_, F, N> where [(); 1 << N]: {}
impl<F: Field + Copy + Debug, const N: usize> Group for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn identity() -> Self {
    CliffordAlgebraElement { value: Vector::<{ 1 << N }, F>::zero(), bilinear_space: None }
  }

  fn inverse(&self) -> Self { Self { value: -self.value, bilinear_space: self.bilinear_space } }
}
impl<F: Field + Copy + Debug, const N: usize> AbelianGroup for CliffordAlgebraElement<'_, F, N> where [(); 1 << N]: {}
impl<F: Field + Copy + Debug + Mul<Self>, const N: usize> LeftModule
  for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Ring = F;
}
impl<F: Field + Copy + Debug + Mul<Self>, const N: usize> RightModule
  for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Ring = F;
}
impl<F: Field + Copy + Debug + Mul<Self>, const N: usize> TwoSidedModule
  for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Ring = F;
}
impl<F: Field + Copy + Debug + Mul<Self>, const N: usize> VectorSpace
  for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
}
impl<F: Field + Copy + Debug + Mul<Self>, const N: usize> Algebra
  for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
}

impl<F: Field + Copy + Display, const N: usize> Display for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    let mut first = true;

    // Helper function to write basis element
    let write_basis = |f: &mut Formatter<'_>, indices: &[usize]| -> std::fmt::Result {
      if indices.is_empty() {
        return Ok(());
      }

      write!(f, "e")?;
      for (i, &index) in indices.iter().enumerate() {
        // Convert number to Unicode subscript by converting each digit
        let num = index;
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
          write!(f, "{subscript}")?;
        }

        if i < indices.len() - 1 {
          write!(f, "‚")?;
        }
      }
      Ok(())
    };

    // Print each grade in order
    for grade in 0..=N {
      // Generate all possible combinations of indices for this grade
      let mut indices = Vec::with_capacity(grade);
      let mut combinations = Vec::new();
      generate_combinations(0, N, grade, &mut indices, &mut combinations);

      // Sort combinations by their bit position
      combinations.sort_by_key(|indices| CliffordAlgebra::<F, N>::blade_indices_to_bit(indices));

      // Print each combination in order
      for indices in combinations {
        let bit_position = CliffordAlgebra::<F, N>::blade_indices_to_bit(&indices);
        if !self.value.0[bit_position].is_zero() {
          if !first {
            write!(f, " + ")?;
          }
          write!(f, "{}", self.value.0[bit_position])?;
          if grade > 0 {
            write_basis(f, &indices)?;
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

/// Helper function to generate all combinations of indices for a given grade
fn generate_combinations(
  start: usize,
  n: usize,
  k: usize,
  current: &mut Vec<usize>,
  result: &mut Vec<Vec<usize>>,
) {
  if k == 0 {
    result.push(current.clone());
    return;
  }

  for i in start..n {
    current.push(i);
    generate_combinations(i + 1, n, k - 1, current, result);
    current.pop();
  }
}

#[macro_export]
macro_rules! impl_mul_scalar_clifford {
  ($($t:ty)*) => ($(
    impl<'a, const N: usize> Mul<CliffordAlgebraElement<'a, $t, N>> for $t
    where [(); 1 << N]:
    {
      type Output = CliffordAlgebraElement<'a, $t, N>;

      fn mul(self, rhs: CliffordAlgebraElement<'a, $t, N>) -> Self::Output { rhs * self }
    }
  )*)
}

impl_mul_scalar_clifford!(f32);
impl_mul_scalar_clifford!(f64);

/// Helper function to multiply two basis blades
fn multiply_blades<F: Field + Copy, const N: usize>(
  left: &[usize],
  right: &[usize],
  bilinear_space: &BilinearSpace<F, N>,
) -> (bool, Vec<usize>) {
  let mut result_indices = Vec::new();
  let mut sign = false;

  // Handle the case where either blade is empty (scalar)
  if left.is_empty() {
    return (true, right.to_vec());
  }
  if right.is_empty() {
    return (true, left.to_vec());
  }

  // Merge the indices while keeping track of sign changes
  let mut i = 0;
  let mut j = 0;

  while i < left.len() && j < right.len() {
    if left[i] == right[j] {
      // Same index: apply quadratic form
      let mut v = Vector::<N, F>::zero();
      v.0[left[i]] = F::one();
      let q = bilinear_space.evaluate(&v, &v);
      sign = !sign;
      i += 1;
      j += 1;
    } else if left[i] < right[j] {
      // Left index comes first: no sign change
      result_indices.push(left[i]);
      i += 1;
    } else {
      // Right index comes first: count swaps for sign
      result_indices.push(right[j]);
      sign = !sign;
      j += 1;
    }
  }

  // Add remaining indices
  result_indices.extend_from_slice(&left[i..]);
  result_indices.extend_from_slice(&right[j..]);

  (sign, result_indices)
}

impl<F: Field + Copy, const N: usize> CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  /// Maps a bit position to the corresponding basis blade indices.
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
      for j in if i == 0 { 0 } else { indices[i - 1] + 1 }..idx {
        remaining_bits += binomial(N - j - 1, grade - i - 1);
      }
    }

    bit_position + remaining_bits
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn clifford_algebra() -> CliffordAlgebra<f64, 3> {
    let bilinear_space = BilinearSpace::new(Vector::<3, f64>([1.0, 1.0, -1.0]));
    CliffordAlgebra::new(bilinear_space)
  }

  #[test]
  fn test_display_order() {
    let algebra = clifford_algebra();
    let one = algebra.element(Vector::<8, f64>([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let e0 = algebra.element(Vector::<8, f64>([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let e1 = algebra.element(Vector::<8, f64>([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let e2 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
    let e01 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    let e02 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
    let e12 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    let e012 = algebra.element(Vector::<8, f64>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));

    let sum = one + 2.0 * e0 + 3.0 * e1 + 4.0 * e2 + 5.0 * e01 + 6.0 * e02 + 7.0 * e12 + 8.0 * e012;
    assert_eq!(format!("{sum}"), "1 + 2e₀ + 3e₁ + 4e₂ + 5e₀‚₁ + 6e₀‚₂ + 7e₁‚₂ + 8e₀‚₁‚₂");
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

  #[test]
  fn test_blade() {
    let algebra = clifford_algebra();

    let e1 = algebra.blade([1]);
    assert_eq!(format!("{e1}"), "1e₁");
  }

  #[test]
  fn test_add() {
    let algebra = clifford_algebra();
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);
    let sum = e1 + e2;
    assert_eq!(format!("{sum}"), "1e₁ + 1e₂");
  }

  #[test]
  fn test_mul() {
    let algebra = clifford_algebra();
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);
    let sum = e1 * e2;
    assert_eq!(format!("{sum}"), "1e₁‚₂");
  }
}
