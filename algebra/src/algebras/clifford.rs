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
pub struct QuadraticForm<F: Field, const N: usize> {
  coefficients: Vector<N, F>,
}

impl<F: Field + Copy, const N: usize> QuadraticForm<F, N> {
  pub const fn new(coefficients: Vector<N, F>) -> Self { Self { coefficients } }

  pub fn evaluate(&self, v: &Vector<N, F>) -> F {
    let mut result = <F as Ring>::zero();
    for i in 0..N {
      result += self.coefficients.0[i] * v.0[i] * v.0[i];
    }
    result
  }
}

pub struct CliffordAlgebra<F: Field, const N: usize> {
  quadratic_form: QuadraticForm<F, N>,
}

impl<F: Field + Copy, const N: usize> CliffordAlgebra<F, N>
where [(); 1 << N]:
{
  pub const fn new(quadratic_form: QuadraticForm<F, N>) -> Self { Self { quadratic_form } }

  pub const fn element(&self, value: Vector<{ 1 << N }, F>) -> CliffordAlgebraElement<'_, F, N> {
    CliffordAlgebraElement { value, quadratic_form: Some(&self.quadratic_form) }
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

    CliffordAlgebraElement { value, quadratic_form: Some(&self.quadratic_form) }
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
  quadratic_form: Option<&'a QuadraticForm<F, N>>,
}

// TODO: All of these impls should check the same bilinear space. This should probably be a compile
// time thing.

impl<F: Field + Copy + Debug, const N: usize> Add for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    assert_eq!(self.quadratic_form, other.quadratic_form);
    Self { value: self.value + other.value, quadratic_form: self.quadratic_form }
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

  fn neg(self) -> Self::Output { Self { value: -self.value, quadratic_form: self.quadratic_form } }
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

impl<F: Field + Copy + Debug, const N: usize> Mul for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    assert_eq!(self.quadratic_form, other.quadratic_form);
    let quadratic_form = self.quadratic_form.expect("Both elements must have a bilinear space");

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
        let (sign, product_indices) = multiply_blades::<F, N>(&left_indices, &right_indices);

        // Calculate the coefficient
        let mut coefficient = self.value.0[i] * other.value.0[j];

        // Apply sign
        coefficient = match sign {
          Sign::Positive => coefficient,
          Sign::Negative => -coefficient,
        };

        // Apply quadratic form for any repeated indices
        let mut repeated_indices = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < left_indices.len() && j < right_indices.len() {
          if left_indices[i] == right_indices[j] {
            repeated_indices.push(left_indices[i]);
            i += 1;
            j += 1;
          } else if left_indices[i] < right_indices[j] {
            i += 1;
          } else {
            j += 1;
          }
        }

        // Multiply by quadratic form coefficients for each repeated index
        for &idx in &repeated_indices {
          coefficient = coefficient * quadratic_form.coefficients.0[idx];
        }

        // Add to the result
        let product_bit = Self::blade_indices_to_bit(&product_indices);
        result.0[product_bit] += coefficient;
      }
    }

    Self { value: result, quadratic_form: self.quadratic_form }
  }
}

impl<F: Field + Copy + Debug, const N: usize> MulAssign for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<F: Field + Copy, const N: usize> Mul<F> for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  type Output = Self;

  fn mul(self, rhs: F) -> Self::Output {
    Self { value: self.value * rhs, quadratic_form: self.quadratic_form }
  }
}

impl<F: Field + Copy + Debug, const N: usize> Multiplicative for CliffordAlgebraElement<'_, F, N> where [(); 1 << N]: {}

// TODO: This is weird... i'll use option to note the zero element.
impl<F: Field + Copy + Debug, const N: usize> Zero for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn zero() -> Self { Self { value: Vector::<{ 1 << N }, F>::zero(), quadratic_form: None } }

  fn is_zero(&self) -> bool { self.value.is_zero() }
}

impl<F: Field + Copy + Debug, const N: usize> Additive for CliffordAlgebraElement<'_, F, N> where [(); 1 << N]: {}
impl<F: Field + Copy + Debug, const N: usize> Group for CliffordAlgebraElement<'_, F, N>
where [(); 1 << N]:
{
  fn identity() -> Self {
    CliffordAlgebraElement { value: Vector::<{ 1 << N }, F>::zero(), quadratic_form: None }
  }

  fn inverse(&self) -> Self { Self { value: -self.value, quadratic_form: self.quadratic_form } }
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

pub enum Sign {
  Positive,
  Negative,
}

/// Helper function to multiply two basis blades
fn multiply_blades<F: Field + Copy, const N: usize>(
  left: &[usize],
  right: &[usize],
) -> (Sign, Vec<usize>) {
  let mut result_indices = Vec::new();
  let mut sign = Sign::Positive;

  // Handle the case where either blade is empty (scalar)
  if left.is_empty() {
    return (Sign::Positive, right.to_vec());
  }
  if right.is_empty() {
    return (Sign::Positive, left.to_vec());
  }

  // Merge the indices while keeping track of sign changes
  let mut i = 0;
  let mut j = 0;

  while i < left.len() && j < right.len() {
    if left[i] == right[j] {
      // Same index: apply quadratic form
      // The quadratic form coefficient will be applied in the multiplication
      // We just need to remove both indices
      i += 1;
      j += 1;
    } else if left[i] < right[j] {
      // Left index comes first: no sign change
      result_indices.push(left[i]);
      i += 1;
    } else {
      // Right index comes first: count swaps for sign
      result_indices.push(right[j]);
      // Each time we move a right index past a left index, we need to count the number
      // of left indices remaining to determine the sign change
      if (left.len() - i) % 2 == 1 {
        sign = match sign {
          Sign::Positive => Sign::Negative,
          Sign::Negative => Sign::Positive,
        };
      }
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

  fn clifford_algebra_non_euclidean() -> CliffordAlgebra<f64, 3> {
    let quadratic_form = QuadraticForm::new(Vector::<3, f64>([1.0, 1.0, -1.0]));
    CliffordAlgebra::new(quadratic_form)
  }

  fn clifford_algebra_euclidean() -> CliffordAlgebra<f64, 3> {
    let quadratic_form = QuadraticForm::new(Vector::<3, f64>([1.0, 1.0, 1.0]));
    CliffordAlgebra::new(quadratic_form)
  }

  #[test]
  fn test_display_order() {
    let algebra = clifford_algebra_non_euclidean();
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
    let algebra = clifford_algebra_non_euclidean();
    let e1 = algebra.blade([1]);
    assert_eq!(format!("{e1}"), "1e₁");
  }

  #[test]
  fn test_add() {
    let algebra = clifford_algebra_non_euclidean();
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);
    let sum = e1 + e2;
    assert_eq!(format!("{sum}"), "1e₁ + 1e₂");
  }

  #[test]
  fn test_mul_basic() {
    let algebra = clifford_algebra_non_euclidean();
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);
    let product = e1 * e2;
    assert_eq!(format!("{product}"), "1e₁‚₂");
  }

  #[test]
  fn test_mul_with_quadratic_form() {
    let algebra = clifford_algebra_non_euclidean();
    let e1 = algebra.blade([1]);
    let e01 = algebra.blade([0, 1]);
    let product = e1 * e01;
    assert_eq!(format!("{product}"), "-1e₀");
  }

  #[test]
  fn test_mul_euclidean() {
    let algebra = clifford_algebra_euclidean();
    let e1 = algebra.blade([1]);
    let e01 = algebra.blade([0, 1]);
    let product = e1 * e01;
    assert_eq!(format!("{product}"), "-1e₀");
  }

  #[test]
  fn test_mul_anti_commutativity() {
    let algebra = clifford_algebra_non_euclidean();
    let e0 = algebra.blade([0]);
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);

    // Test anti-commutativity of basis vectors
    assert_eq!(format!("{}", e0 * e1), "1e₀‚₁");
    assert_eq!(format!("{}", e1 * e0), "-1e₀‚₁");
    assert_eq!(format!("{}", e1 * e2), "1e₁‚₂");
    assert_eq!(format!("{}", e2 * e1), "-1e₁‚₂");
  }

  #[test]
  fn test_mul_scalar() {
    let algebra = clifford_algebra_non_euclidean();
    let one = algebra.element(Vector::<8, f64>([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);

    // Test scalar multiplication
    assert_eq!(format!("{}", one * e1), "1e₁");
    assert_eq!(format!("{}", e1 * one), "1e₁");
    assert_eq!(format!("{}", one * e2), "1e₂");
    assert_eq!(format!("{}", e2 * one), "1e₂");
  }

  #[test]
  fn test_mul_quadratic_form_application() {
    let algebra = clifford_algebra_non_euclidean();
    let e0 = algebra.blade([0]);
    let e1 = algebra.blade([1]);
    let e2 = algebra.blade([2]);

    // Test quadratic form application
    assert_eq!(format!("{}", e0 * e0), "1"); // Q(e0) = 1
    assert_eq!(format!("{}", e1 * e1), "1"); // Q(e1) = 1
    assert_eq!(format!("{}", e2 * e2), "-1"); // Q(e2) = -1
  }

  #[test]
  fn test_mul_higher_grade() {
    let algebra = clifford_algebra_non_euclidean();
    let e01 = algebra.blade([0, 1]);
    let e12 = algebra.blade([1, 2]);
    let e02 = algebra.blade([0, 2]);

    // Test multiplication of bivectors
    assert_eq!(format!("{}", e01 * e12), "1e₀‚₂"); // e01 * e12 = e0 * e1 * e1 * e2 = e0 * Q(e1) * e2 = e0 * 1 * e2 = e0 * e2 = e02

    assert_eq!(format!("{}", e12 * e01), "1e₀‚₂");
    assert_eq!(format!("{}", e02 * e12), "1e₀‚₁"); // e02 * e12 = e0 * e2 * e1 * e2 = -e0 * e1 * e2
                                                   // * e2 = -e0 * e1 * Q(e2) = -e0 * e1 * -1 = e0
                                                   // * e1 = e01
  }

  #[test]
  fn test_mul_trivector() {
    let algebra = clifford_algebra_non_euclidean();
    let e01 = algebra.blade([0, 1]);
    let e2 = algebra.blade([2]);
    let e012 = algebra.blade([0, 1, 2]);

    // Test multiplication with trivector
    assert_eq!(format!("{}", e01 * e2), "1e₀‚₁‚₂");
    assert_eq!(format!("{}", e2 * e01), "1e₀‚₁‚₂");
    assert_eq!(format!("{}", e012 * e2), "-1e₀‚₁"); // e012 * e2 = e0 * e1 * e2 * e2 = e0 * e1 *
                                                    // Q(e2) = e0 * e1 * -1 = -e0 * e1 = -e01
  }
}
