//! Implementation of tropical algebra, sometimes called max-plus algebra.
//!
//! A tropical algebra is a semiring where addition is replaced by maximum
//! and multiplication is replaced by addition. The tropical semiring is defined as:
//! (ℝ ∪ {-∞}, max, +)
//!
//! In this implementation, we use the max-plus variant where:
//! - Addition (⊕) is defined as maximum: a ⊕ b = max(a, b)
//! - Multiplication (⊗) is defined as addition: a ⊗ b = a + b
//! - Zero is -∞ (f64::NEG_INFINITY)
//! - One is 0
//!
//! Tropical algebra is useful in various applications including:
//! - Optimization problems
//! - Scheduling
//! - Network analysis
//! - Discrete event systems
//! - Algebraic geometry
//!
//! # Examples
//!
//! ```
//! use harness_algebra::semimodule::tropical::{BilinearForm, TropicalAlgebra, TropicalElement};
//!
//! // Create tropical elements
//! let a = TropicalElement::new(3.0);
//! let b = TropicalElement::new(5.0);
//!
//! // Addition is max: 3 ⊕ 5 = 5
//! let sum = a + b;
//! assert_eq!(sum.value(), 5.0);
//!
//! // Multiplication is +: 3 ⊗ 5 = 8
//! let product = a * b;
//! assert_eq!(product.value(), 8.0);
//!
//! // Create a tropical algebra with a bilinear form
//! let matrix = [[TropicalElement::new(1.0), TropicalElement::new(2.0)], [
//!   TropicalElement::new(2.0),
//!   TropicalElement::new(1.0),
//! ]];
//! let bilinear_form = BilinearForm::new(matrix);
//! let algebra = TropicalAlgebra::new(bilinear_form);
//!
//! // Evaluate the bilinear form on vectors
//! let x = [TropicalElement::new(3.0), TropicalElement::new(4.0)];
//! let y = [TropicalElement::new(5.0), TropicalElement::new(6.0)];
//! let result = algebra.evaluate(&x, &y);
//! assert_eq!(result.value(), 11.0);
//! ```

use std::fmt::Debug;

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, MulAssign, Multiplicative, One, Zero},
  ring::Semiring,
};

/// Trait for types that have a concept of negative infinity.
pub trait NegInfinity {
  /// Returns the negative infinity value for the type.
  fn neg_infinity() -> Self;
}

impl NegInfinity for f64 {
  fn neg_infinity() -> Self { f64::NEG_INFINITY }
}

impl NegInfinity for f32 {
  fn neg_infinity() -> Self { f32::NEG_INFINITY }
}

impl NegInfinity for i32 {
  fn neg_infinity() -> Self { i32::MIN }
}

impl NegInfinity for i64 {
  fn neg_infinity() -> Self { i64::MIN }
}

impl NegInfinity for i128 {
  fn neg_infinity() -> Self { i128::MIN }
}

impl NegInfinity for i8 {
  fn neg_infinity() -> Self { i8::MIN }
}

impl NegInfinity for i16 {
  fn neg_infinity() -> Self { i16::MIN }
}

impl NegInfinity for u8 {
  fn neg_infinity() -> Self { u8::MIN }
}

impl NegInfinity for u16 {
  fn neg_infinity() -> Self { u16::MIN }
}

impl NegInfinity for u32 {
  fn neg_infinity() -> Self { u32::MIN }
}

impl NegInfinity for u64 {
  fn neg_infinity() -> Self { u64::MIN }
}

impl NegInfinity for u128 {
  fn neg_infinity() -> Self { u128::MIN }
}

/// An element of the tropical algebra.
///
/// In tropical algebra:
/// - Addition (⊕) is defined as maximum: a ⊕ b = max(a, b)
/// - Multiplication (⊗) is defined as addition: a ⊗ b = a + b
/// - Zero is -∞ (f64::NEG_INFINITY)
/// - One is 0
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity {
  value: F,
}

impl<F> Eq for TropicalElement<F> where F: Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
}
impl<F> TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  /// Creates a new tropical element with the given value.
  /// Panics if the value is NaN or infinite.
  pub fn new(value: F) -> Self { Self { value } }

  /// Returns the value of this tropical element.
  pub fn value(&self) -> F { self.value }
}

impl<F> Add for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    // In tropical algebra, addition is maximum
    Self { value: if self.value > other.value { self.value } else { other.value } }
  }
}

impl<F> AddAssign for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<F> Mul for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    // In tropical algebra, multiplication is addition
    #[allow(clippy::suspicious_arithmetic_impl)]
    Self { value: self.value + other.value }
  }
}

impl<F> MulAssign for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<F> Zero for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  fn zero() -> Self { Self { value: F::neg_infinity() } }

  fn is_zero(&self) -> bool { self.value == F::neg_infinity() }
}

impl<F> One for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  fn one() -> Self { Self { value: F::zero() } }
}

impl<F> Additive for TropicalElement<F> where F: Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
}
impl<F> Multiplicative for TropicalElement<F> where F: Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
}
impl<F> Semiring for TropicalElement<F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  fn zero() -> Self { Self { value: F::neg_infinity() } }

  fn one() -> Self { Self { value: F::zero() } }
}

/// Symmetric bilinear form
#[derive(Debug, PartialEq, Eq)]
pub struct BilinearForm<const N: usize, F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity {
  matrix: [[TropicalElement<F>; N]; N],
}

impl<const N: usize, F> BilinearForm<N, F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  /// Creates a new bilinear form with the given matrix.
  pub const fn new(matrix: [[TropicalElement<F>; N]; N]) -> Self { Self { matrix } }

  /// Evaluates the bilinear form on two vectors.
  pub fn evaluate(
    &self,
    x: &[TropicalElement<F>; N],
    y: &[TropicalElement<F>; N],
  ) -> TropicalElement<F> {
    let mut result = <TropicalElement<F> as Semiring>::zero();

    for (i, &xi) in x.iter().enumerate() {
      for (j, &yj) in y.iter().enumerate() {
        let term = xi * self.matrix[i][j] * yj;
        result = if term > result { term } else { result };
      }
    }

    result
  }
}

/// A tropical algebra.
pub struct TropicalAlgebra<const N: usize, F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity {
  /// The bilinear form defining the algebra
  bilinear_form: BilinearForm<N, F>,
}

impl<const N: usize, F> TropicalAlgebra<N, F>
where F: Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + NegInfinity
{
  /// Creates a new tropical algebra with the given bilinear form.
  pub const fn new(bilinear_form: BilinearForm<N, F>) -> Self { Self { bilinear_form } }

  /// Evaluates the bilinear form on two vectors.
  pub fn evaluate(
    &self,
    x: &[TropicalElement<F>; N],
    y: &[TropicalElement<F>; N],
  ) -> TropicalElement<F> {
    self.bilinear_form.evaluate(x, y)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_tropical_element_operations() {
    let a = TropicalElement::new(3.0);
    let b = TropicalElement::new(5.0);
    let c = TropicalElement::new(2.0);

    // Test addition (max)
    assert_eq!(a + b, TropicalElement::new(5.0));
    assert_eq!(a + c, TropicalElement::new(3.0));
    assert_eq!(b + c, TropicalElement::new(5.0));

    // Test multiplication (addition)
    assert_eq!(a * b, TropicalElement::new(8.0));
    assert_eq!(a * c, TropicalElement::new(5.0));
    assert_eq!(b * c, TropicalElement::new(7.0));

    // Test zero and one
    assert_eq!(<TropicalElement<f64> as Semiring>::zero().value(), f64::NEG_INFINITY);
    assert_eq!(<TropicalElement<f64> as Semiring>::one().value(), 0.0);

    // Test additive identity
    assert_eq!(a + <TropicalElement<f64> as Semiring>::zero(), a);
    assert_eq!(<TropicalElement<f64> as Semiring>::zero() + a, a);

    // Test multiplicative identity
    assert_eq!(a * <TropicalElement<f64> as Semiring>::one(), a);
    assert_eq!(<TropicalElement<f64> as Semiring>::one() * a, a);

    // Test additive assignment
    let mut x = a;
    x += b;
    assert_eq!(x, TropicalElement::new(5.0));

    // Test multiplicative assignment
    let mut y = a;
    y *= b;
    assert_eq!(y, TropicalElement::new(8.0));
  }

  #[test]
  fn test_bilinear_form_evaluation() {
    let matrix = [[TropicalElement::new(1.0), TropicalElement::new(2.0)], [
      TropicalElement::new(2.0),
      TropicalElement::new(1.0),
    ]];
    let bilinear_form = BilinearForm::new(matrix);

    // Test with simple vectors
    let x = [TropicalElement::new(3.0), TropicalElement::new(4.0)];
    let y = [TropicalElement::new(5.0), TropicalElement::new(6.0)];

    // B(x,y) = max(x₁ + M₁₁ + y₁, x₁ + M₁₂ + y₂, x₂ + M₂₁ + y₁, x₂ + M₂₂ + y₂)
    // = max(3 + 1 + 5, 3 + 2 + 6, 4 + 2 + 5, 4 + 1 + 6)
    // = max(9, 11, 11, 11)
    // = 11
    assert_eq!(bilinear_form.evaluate(&x, &y), TropicalElement::new(11.0));

    // Test with zero vector
    let zero =
      [<TropicalElement<f64> as Semiring>::zero(), <TropicalElement<f64> as Semiring>::zero()];
    assert_eq!(bilinear_form.evaluate(&zero, &y), <TropicalElement<f64> as Semiring>::zero());
    assert_eq!(bilinear_form.evaluate(&x, &zero), <TropicalElement<f64> as Semiring>::zero());

    // Test with one vector
    let one =
      [<TropicalElement<f64> as Semiring>::one(), <TropicalElement<f64> as Semiring>::one()];
    assert_eq!(bilinear_form.evaluate(&one, &one), TropicalElement::new(2.0));
  }

  #[test]
  fn test_tropical_algebra() {
    let matrix = [[TropicalElement::new(1.0), TropicalElement::new(2.0)], [
      TropicalElement::new(2.0),
      TropicalElement::new(1.0),
    ]];
    let bilinear_form = BilinearForm::new(matrix);
    let algebra = TropicalAlgebra::new(bilinear_form);

    // Test evaluation through algebra
    let x = [TropicalElement::new(3.0), TropicalElement::new(4.0)];
    let y = [TropicalElement::new(5.0), TropicalElement::new(6.0)];
    assert_eq!(algebra.evaluate(&x, &y), TropicalElement::new(11.0));

    // Test with different vectors
    let a = [TropicalElement::new(0.0), TropicalElement::new(1.0)];
    let b = [TropicalElement::new(2.0), TropicalElement::new(3.0)];
    assert_eq!(algebra.evaluate(&a, &b), TropicalElement::new(5.0));
  }

  #[test]
  fn test_tropical_element_ordering() {
    let a = TropicalElement::new(3.0);
    let b = TropicalElement::new(5.0);
    let c = TropicalElement::new(3.0);

    assert!(a < b);
    assert!(b > a);
    assert!(a <= c);
    assert!(a >= c);
    assert_eq!(a, c);
  }

  #[test]
  fn test_tropical_element_zero_one_properties() {
    let a = TropicalElement::new(3.0);
    let zero = <TropicalElement<f64> as Semiring>::zero();
    let one = <TropicalElement<f64> as Semiring>::one();

    // Test zero properties
    assert!(zero.is_zero());
    assert_eq!(a + zero, a);
    assert_eq!(zero + a, a);
    assert_eq!(a * zero, zero);
    assert_eq!(zero * a, zero);

    // Test one properties
    assert_eq!(a * one, a);
    assert_eq!(one * a, a);
  }
}
