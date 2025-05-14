//! Tropical Polynomial Module
//!
//! This module implements tropical polynomials over the max-plus semiring.
//! In tropical algebra, addition is defined as the maximum of two numbers,
//! and multiplication is defined as the sum of two numbers.
//!
//! # Examples
//!
//! ```
//! use std::collections::HashMap;
//!
//! use harness_algebra::{
//!   algebras::tropical_poly::TropicalPolynomial, semimodule::tropical::TropicalElement,
//! };
//!
//! // Create p(x) = 2⊗x ⊕ 3
//! let mut terms = HashMap::new();
//! terms.insert(vec![1], TropicalElement::new(2.0)); // 2⊗x
//! terms.insert(vec![0], TropicalElement::new(3.0)); // 3
//! let p = TropicalPolynomial::new(terms);
//! ```

use std::collections::HashMap;

use crate::{
  arithmetic::{Add, Mul, One, Zero},
  semimodule::tropical::TropicalElement,
};

/// A tropical polynomial over the max-plus semiring.
///
/// A tropical polynomial is a polynomial where addition is the maximum operation
/// and multiplication is addition in the underlying field.
///
/// # Type Parameters
///
/// * `F` - The underlying field type, must implement the required traits for tropical arithmetic
///   operations.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use harness_algebra::{
///   algebras::tropical_poly::TropicalPolynomial, semimodule::tropical::TropicalElement,
/// };
///
/// // Create p(x) = 2⊗x ⊕ 3
/// let mut terms = HashMap::new();
/// terms.insert(vec![1], TropicalElement::new(2.0)); // 2⊗x
/// terms.insert(vec![0], TropicalElement::new(3.0)); // 3
/// let p = TropicalPolynomial::new(terms);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TropicalPolynomial<F>
where F: Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + std::fmt::Debug {
  /// The terms of the polynomial, mapping from exponents to coefficients.
  /// Each key is a vector of exponents for each variable, and each value
  /// is the tropical coefficient for that term.
  terms: HashMap<Vec<usize>, TropicalElement<F>>,
}

impl<F> TropicalPolynomial<F>
where F: Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + std::fmt::Debug
{
  /// Creates a new tropical polynomial from a map of terms.
  ///
  /// # Arguments
  ///
  /// * `terms` - A HashMap mapping from exponent vectors to tropical coefficients.
  ///
  /// # Returns
  ///
  /// A new `TropicalPolynomial` with the given terms.
  pub fn new(terms: HashMap<Vec<usize>, TropicalElement<F>>) -> Self { Self { terms } }

  /// Returns the degree of the polynomial.
  ///
  /// The degree is the maximum sum of exponents across all terms.
  /// For example, in a term like x²y³, the degree would be 5.
  ///
  /// # Returns
  ///
  /// The degree of the polynomial, or 0 if the polynomial is empty.
  pub fn degree(&self) -> usize {
    self.terms.keys().map(|exponents| exponents.iter().sum()).max().unwrap_or(0)
  }
}

/// Implementation of tropical polynomial addition.
///
/// In tropical algebra, addition is the maximum operation.
/// When adding two polynomials, terms with the same exponents
/// take the maximum of their coefficients.
impl<F> Add for TropicalPolynomial<F>
where F: Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + std::fmt::Debug
{
  type Output = Self;

  fn add(self, other: Self) -> Self {
    let mut result = self.terms;

    // For each term in the other polynomial
    for (exponents, coefficient) in other.terms {
      // Use tropical addition (maximum) for terms with the same exponents
      result
                .entry(exponents)
                .and_modify(|e| *e += coefficient) // This uses TropicalElement's add which is max
                .or_insert(coefficient);
    }

    Self::new(result)
  }
}

/// Implementation of tropical polynomial multiplication.
///
/// In tropical algebra, multiplication is addition in the underlying field.
/// When multiplying two polynomials:
/// 1. Exponents are added for each variable
/// 2. Coefficients are added (since tropical multiplication is addition)
/// 3. Terms with the same exponents take the maximum of their coefficients
impl<F> Mul for TropicalPolynomial<F>
where F: Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = F>
    + Mul<Output = F>
    + Zero
    + One
    + std::fmt::Debug
{
  type Output = Self;

  fn mul(self, other: Self) -> Self {
    // If either polynomial is the zero polynomial (empty), return the other polynomial
    if self.terms.is_empty() {
      return other;
    }
    if other.terms.is_empty() {
      return self;
    }

    let mut result = HashMap::new();

    // For each term in the first polynomial
    for (exponents1, coefficient1) in &self.terms {
      // For each term in the second polynomial
      for (exponents2, coefficient2) in &other.terms {
        // Add the exponents for each variable
        let new_exponents: Vec<usize> =
          exponents1.iter().zip(exponents2.iter()).map(|(a, b)| a + b).collect();

        // Multiply the coefficients (which is addition in tropical algebra)
        let new_coefficient = *coefficient1 * *coefficient2;

        // Add this term to the result (taking maximum for same exponents)
        result
          .entry(new_exponents)
          .and_modify(|e| *e += new_coefficient)
          .or_insert(new_coefficient);
      }
    }

    Self::new(result)
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashMap;

  use super::*;

  #[test]
  fn test_tropical_polynomial_addition() {
    // Create p(x) = 3⊗x ⊕ 2⊗x²
    let mut p_terms = HashMap::new();
    p_terms.insert(vec![1], TropicalElement::new(3.0)); // 3⊗x
    p_terms.insert(vec![2], TropicalElement::new(2.0)); // 2⊗x²
    let p = TropicalPolynomial::new(p_terms);

    // Create q(x) = 5⊗x ⊕ 1⊗x² ⊕ 4⊗x³
    let mut q_terms = HashMap::new();
    q_terms.insert(vec![1], TropicalElement::new(5.0)); // 5⊗x
    q_terms.insert(vec![2], TropicalElement::new(1.0)); // 1⊗x²
    q_terms.insert(vec![3], TropicalElement::new(4.0)); // 4⊗x³
    let q = TropicalPolynomial::new(q_terms);

    // p(x) ⊕ q(x) should be:
    // - For x: max(3, 5) = 5
    // - For x²: max(2, 1) = 2
    // - For x³: 4 (only in q)
    let sum = p + q;

    let mut expected_terms = HashMap::new();
    expected_terms.insert(vec![1], TropicalElement::new(5.0)); // max(3, 5) = 5
    expected_terms.insert(vec![2], TropicalElement::new(2.0)); // max(2, 1) = 2
    expected_terms.insert(vec![3], TropicalElement::new(4.0)); // from q only
    let expected = TropicalPolynomial::new(expected_terms);

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_tropical_polynomial_multiplication() {
    // Create p(x) = 2⊗x ⊕ 3
    let mut p_terms = HashMap::new();
    p_terms.insert(vec![1], TropicalElement::new(2.0)); // 2⊗x
    p_terms.insert(vec![0], TropicalElement::new(3.0)); // 3
    let p = TropicalPolynomial::new(p_terms);

    // Create q(x) = 1⊗x ⊕ 4
    let mut q_terms = HashMap::new();
    q_terms.insert(vec![1], TropicalElement::new(1.0)); // 1⊗x
    q_terms.insert(vec![0], TropicalElement::new(4.0)); // 4
    let q = TropicalPolynomial::new(q_terms);

    // p(x) ⊗ q(x) should be:
    // (2⊗x ⊕ 3) ⊗ (1⊗x ⊕ 4)
    // = (2⊗x ⊗ 1⊗x) ⊕ (2⊗x ⊗ 4) ⊕ (3 ⊗ 1⊗x) ⊕ (3 ⊗ 4)
    // = (3⊗x²) ⊕ (6⊗x) ⊕ (4⊗x) ⊕ (7)
    // = (3⊗x²) ⊕ (6⊗x) ⊕ (7)
    let product = p * q;

    let mut expected_terms = HashMap::new();
    expected_terms.insert(vec![2], TropicalElement::new(3.0)); // 3⊗x²
    expected_terms.insert(vec![1], TropicalElement::new(6.0)); // 6⊗x
    expected_terms.insert(vec![0], TropicalElement::new(7.0)); // 7
    let expected = TropicalPolynomial::new(expected_terms);

    assert_eq!(product, expected);
  }

  #[test]
  fn test_zero_polynomial_multiplication() {
    // Create zero polynomial
    let zero = TropicalPolynomial::<f64>::new(HashMap::new());

    // Create a non-zero polynomial p(x) = 2⊗x ⊕ 3
    let mut p_terms = HashMap::new();
    p_terms.insert(vec![1], TropicalElement::new(2.0)); // 2⊗x
    p_terms.insert(vec![0], TropicalElement::new(3.0)); // 3
    let p = TropicalPolynomial::new(p_terms);

    // Test zero * p = p
    assert_eq!(zero.clone() * p.clone(), p);

    // Test p * zero = p
    assert_eq!(p.clone() * zero.clone(), p.clone());
  }

  #[test]
  fn test_section_2_1_example1() {
    // Example 1: p(x, y, z) = 5⊗x⊗y⊗z ⊕ x⊗x ⊕ 2⊗z ⊕ 17
    let mut p_terms = HashMap::new();

    // 5⊗x⊗y⊗z: exponents [1,1,1] for x,y,z
    p_terms.insert(vec![1, 1, 1], TropicalElement::new(5.0));

    // x⊗x: exponents [2,0,0] for x²
    p_terms.insert(vec![2, 0, 0], TropicalElement::new(1.0));

    // 2⊗z: exponents [0,0,1] for z
    p_terms.insert(vec![0, 0, 1], TropicalElement::new(2.0));

    // 17: exponents [0,0,0] for constant term
    p_terms.insert(vec![0, 0, 0], TropicalElement::new(17.0));

    let p = TropicalPolynomial::new(p_terms);
    assert_eq!(p.degree(), 3); // Degree is 3 because of the x⊗y⊗z term
  }

  #[test]
  fn test_section_2_1_example2() {
    // Example 2: p(x) = (0⊗x)⊕(0⊗x⊗x)
    let mut p_terms = HashMap::new();
    p_terms.insert(vec![1], TropicalElement::new(0.0)); // 0⊗x
    p_terms.insert(vec![2], TropicalElement::new(0.0)); // 0⊗x⊗x
    let p = TropicalPolynomial::new(p_terms);

    // Create q(x) = 0 (the zero polynomial)
    let q = TropicalPolynomial::<f64>::new(HashMap::new());

    // Create r(x) = 2⊗x
    let mut r_terms = HashMap::new();
    r_terms.insert(vec![1], TropicalElement::new(2.0));
    let r = TropicalPolynomial::new(r_terms);

    // Verify that p and q are different
    assert_ne!(p, q);

    // Test p(x) ⊗ r(x) = (2⊗x⊗x) ⊕ (2⊗x⊗x⊗x)
    let p_times_r = p * r.clone();
    let mut expected_p_times_r = HashMap::new();
    expected_p_times_r.insert(vec![2], TropicalElement::new(2.0)); // 2⊗x⊗x
    expected_p_times_r.insert(vec![3], TropicalElement::new(2.0)); // 2⊗x⊗x⊗x
    assert_eq!(p_times_r, TropicalPolynomial::new(expected_p_times_r));
    assert_ne!(p_times_r, r.clone()); // Verify it's not equal to r(x)

    // Test q(x) ⊗ r(x) = r(x)
    let q_times_r = q * r.clone();
    assert_eq!(q_times_r, r.clone()); // Zero polynomial times anything should give the same
                                      // polynomial
  }

  // This example differs from the paper, because they use min plus algebra, but we use max plus
  // algebra
  #[test]
  fn test_section_2_1_example3() {
    // Create p(x) = (2⊗x) ⊕ (3⊗x⊗x)
    let mut p_terms = HashMap::new();
    p_terms.insert(vec![1], TropicalElement::new(2.0)); // 2⊗x
    p_terms.insert(vec![2], TropicalElement::new(3.0)); // 3⊗x⊗x
    let p = TropicalPolynomial::new(p_terms);

    // Create q(x) = 5 ⊕ (1⊗x)
    let mut q_terms = HashMap::new();
    q_terms.insert(vec![0], TropicalElement::new(5.0)); // 5
    q_terms.insert(vec![1], TropicalElement::new(1.0)); // 1⊗x
    let q = TropicalPolynomial::new(q_terms);

    // p(x)⊗q(x) should be:
    // [(2⊗x)⊗5]⊕[(2⊗x)⊗(1⊗x)]⊕[(3⊗x⊗x)⊗5]⊕[(3⊗x⊗x)⊗(1⊗x)]
    // = (7⊗x)⊕(3⊗x⊗x)⊕(8⊗x⊗x)⊕(4⊗x⊗x⊗x)
    // = (7⊗x)⊕(8⊗x⊗x)⊕(4⊗x⊗x⊗x)
    let product = p * q;

    let mut expected_terms = HashMap::new();
    expected_terms.insert(vec![1], TropicalElement::new(7.0)); // 7⊗x
    expected_terms.insert(vec![2], TropicalElement::new(8.0)); // 8⊗x⊗x (max of 3 and 8)
    expected_terms.insert(vec![3], TropicalElement::new(4.0)); // 4⊗x⊗x⊗x
    let expected = TropicalPolynomial::new(expected_terms);

    assert_eq!(product, expected);
  }
}
