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
//! ```rust, ignore
//! use cova_algebra::semimodule::tropical::{BilinearForm, TropicalAlgebra, TropicalElement};
//!
//! // Create tropical elements
//! let a = TropicalElement::new(3.0);
//! let b = TropicalElement::new(5.0);
//!
//! // Addition is max: 3 ⊕ 5 = 5
//! let sum = a + b;
//! assert_eq!(sum.value(), TropicalElement::Element(5.0));
//!
//! // Multiplication is +: 3 ⊗ 5 = 8
//! let product = a * b;
//! assert_eq!(product.value(), TropicalElement::Element(8.0));
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
//! assert_eq!(result.value(), TropicalElement::Element(11.0));
//! ```

use std::fmt::Debug;

use super::*;
use crate::rings::Semiring;

/// An element of the tropical algebra.
///
/// In tropical algebra:
/// - Addition (⊕) is defined as maximum: a ⊕ b = max(a, b)
/// - Multiplication (⊗) is defined as addition: a ⊗ b = a + b
/// - Zero is -∞
/// - One is 0
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum TropicalElement<F: Field> {
  /// A finite element of the tropical algebra.
  Element(F),
  /// Represents negative infinity, the zero element in tropical algebra.
  NegInfinity,
}

impl<F: Field> Eq for TropicalElement<F> {}
impl<F: Field> TropicalElement<F> {
  /// Creates a new tropical element with the given value.
  pub fn new(value: F) -> Self { TropicalElement::Element(value) }

  /// Returns the value of this tropical element.
  pub fn value(&self) -> TropicalElement<F> { *self }
}

impl<F: Field + PartialOrd> Add for TropicalElement<F> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    match self {
      TropicalElement::Element(a) => {
        match other {
          TropicalElement::Element(b) => {
            // In tropical algebra, addition is maximum
            Self::Element(if a > b { a } else { b })
          },
          TropicalElement::NegInfinity => Self::Element(a),
        }
      },
      TropicalElement::NegInfinity => match other {
        TropicalElement::Element(b) => Self::Element(b),
        TropicalElement::NegInfinity => Self::NegInfinity,
      },
    }
  }
}

impl<F: Field + PartialOrd> AddAssign for TropicalElement<F> {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<F: Field + PartialOrd> Mul for TropicalElement<F> {
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    match self {
      TropicalElement::Element(a) => {
        match other {
          TropicalElement::Element(b) => {
            // In tropical algebra, multiplication is addition
            #[allow(clippy::suspicious_arithmetic_impl)]
            Self::Element(a + b)
          },
          TropicalElement::NegInfinity => Self::NegInfinity,
        }
      },
      TropicalElement::NegInfinity => Self::NegInfinity,
    }
  }
}

impl<F: Field + PartialOrd> MulAssign for TropicalElement<F> {
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<F: Field + PartialOrd> Zero for TropicalElement<F> {
  fn zero() -> Self { TropicalElement::NegInfinity }

  fn is_zero(&self) -> bool { *self == TropicalElement::NegInfinity }
}

impl<F: Field + PartialOrd> One for TropicalElement<F> {
  fn one() -> Self { TropicalElement::Element(F::zero()) }
}

impl<F: Field + PartialOrd> Additive for TropicalElement<F> {}
impl<F: Field + PartialOrd> Multiplicative for TropicalElement<F> {}
impl<F: Field + PartialOrd> Semiring for TropicalElement<F> {}

/// Symmetric bilinear form
#[derive(Debug, PartialEq, Eq)]
pub struct BilinearForm<const N: usize, F>
where
  F: Field + PartialOrd,
  [(); N * (N + 1) / 2]:, {
  // Store only the upper triangular part of the matrix
  // The elements are stored in row-major order, only including elements where i <= j
  matrix: [TropicalElement<F>; N * (N + 1) / 2],
}

impl<const N: usize, F> BilinearForm<N, F>
where
  F: Field + PartialOrd,
  [(); N * (N + 1) / 2]:, // Ensure the array size is valid at compile time
{
  /// Creates a new bilinear form with the given matrix.
  /// The input matrix should be symmetric.
  pub fn new(matrix: [[TropicalElement<F>; N]; N]) -> Self {
    let mut upper_triangular = [TropicalElement::NegInfinity; N * (N + 1) / 2];
    let mut idx = 0;
    for (i, row) in matrix.iter().enumerate() {
      for (_j, &element) in row.iter().enumerate().skip(i) {
        upper_triangular[idx] = element;
        idx += 1;
      }
    }
    Self { matrix: upper_triangular }
  }

  /// Gets the element at position (i,j) in the matrix.
  /// Since the matrix is symmetric, we only store the upper triangular part.
  fn get(&self, i: usize, j: usize) -> TropicalElement<F> {
    if i <= j {
      // For upper triangular part, use the stored value
      // Calculate index in the flattened upper triangular matrix:
      // idx = (i * (2N - i + 1)) / 2 + (j - i)
      let n = N;
      let idx = i
        .checked_mul(2 * n - i + 1)
        .and_then(|x| x.checked_div(2))
        .and_then(|x| x.checked_add(j - i))
        .expect("Index calculation overflow");
      self.matrix[idx]
    } else {
      // For lower triangular part, use symmetry
      self.get(j, i)
    }
  }

  /// Evaluates the bilinear form on two vectors.
  pub fn evaluate(
    &self,
    x: &[TropicalElement<F>; N],
    y: &[TropicalElement<F>; N],
  ) -> TropicalElement<F> {
    let mut result = TropicalElement::<F>::zero();

    for (i, &xi) in x.iter().enumerate() {
      for (j, &yj) in y.iter().enumerate() {
        let term = xi * self.get(i, j) * yj;
        // In tropical algebra, we want to take the maximum
        // If result is NegInfinity, we should always take the term
        // Otherwise, take the maximum of term and result
        result = match (term, result) {
          (TropicalElement::Element(_), TropicalElement::NegInfinity) => term,
          (TropicalElement::NegInfinity, TropicalElement::Element(_)) => result,
          (TropicalElement::Element(t), TropicalElement::Element(r)) =>
            if t > r {
              term
            } else {
              result
            },
          (TropicalElement::NegInfinity, TropicalElement::NegInfinity) => result,
        };
      }
    }

    result
  }
}

/// A tropical algebra.
pub struct TropicalAlgebra<const N: usize, F>
where
  F: Field + PartialOrd,
  [(); N * (N + 1) / 2]:, {
  /// The bilinear form defining the algebra
  bilinear_form: BilinearForm<N, F>,
}

impl<const N: usize, F> TropicalAlgebra<N, F>
where
  F: Field + PartialOrd,
  [(); N * (N + 1) / 2]:,
{
  /// Creates a new tropical algebra with the given bilinear form.
  pub fn new(bilinear_form: BilinearForm<N, F>) -> Self { Self { bilinear_form } }

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
    assert_eq!(TropicalElement::<f64>::zero().value(), TropicalElement::NegInfinity);
    assert_eq!(TropicalElement::<f64>::one().value(), TropicalElement::Element(0.0));

    // Test additive identity
    assert_eq!(a + TropicalElement::<f64>::zero(), a);
    assert_eq!(TropicalElement::<f64>::zero() + a, a);

    // Test multiplicative identity
    assert_eq!(a * TropicalElement::<f64>::one(), a);
    assert_eq!(TropicalElement::<f64>::one() * a, a);

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
    let zero = [TropicalElement::<f64>::zero(), TropicalElement::<f64>::zero()];
    assert_eq!(bilinear_form.evaluate(&zero, &y), TropicalElement::<f64>::zero());
    assert_eq!(bilinear_form.evaluate(&x, &zero), TropicalElement::<f64>::zero());

    // Test with one vector
    let one = [TropicalElement::<f64>::one(), TropicalElement::<f64>::one()];
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
    let zero = TropicalElement::<f64>::zero();
    let one = TropicalElement::<f64>::one();

    // Test zero properties
    assert!(zero.is_zero());
    dbg!(a);
    dbg!(zero);
    assert_eq!(a + zero, a);
    assert_eq!(zero + a, a);
    assert_eq!(a * zero, zero);
    assert_eq!(zero * a, zero);

    // Test one properties
    assert_eq!(a * one, a);
    assert_eq!(one * a, a);
  }
}
