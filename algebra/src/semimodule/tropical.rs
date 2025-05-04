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
//! use harness_algebra::algebras::tropical::TropicalAlgebra;
//!
//! let algebra = TropicalAlgebra::new();
//! let a = algebra.element(3.0);
//! let b = algebra.element(5.0);
//!
//! // Addition is max: 3 ⊕ 5 = 5
//! let sum = a + b;
//! assert_eq!(sum.value(), 5.0);
//!
//! // Multiplication is +: 3 ⊗ 5 = 8
//! let product = a * b;
//! assert_eq!(product.value(), 8.0);
//! ```

use std::fmt::{Debug, Display, Formatter};

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, MulAssign, Multiplicative, One, Zero},
  ring::Semiring,
};

/// Symmetric bilinear form:  a_ij = a_ji
#[derive(Debug, PartialEq, Eq)]
pub struct BilinearForm<F: Semiring, const N: usize> {
  matrix: [[F; N]; N],
}

impl<F: Semiring + Copy, const N: usize> BilinearForm<F, N> {
  /// Creates a new quadratic form with the given coefficients.
  ///
  /// # Arguments
  ///
  /// * `coefficients` - A vector of coefficients for the quadratic form
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_algebra::{semimodule::tropical::BilinearForm, vector::Vector};
  ///
  /// let q = BilinearForm::new(Vector::<3, f64>([1.0, 1.0, -1.0]));
  /// ```
  pub const fn new(matrix: [[F; N]; N]) -> Self { Self { matrix } }

  //
  pub fn evaluate(&self, v: &Vector<N, F>) -> F {
    let mut result = <F as Semiring>::zero();
    for i in 0..N {
      result += self.matrix[i][i] * v.0[i] * v.0[i];
    }
    result
  }
}

/// A tropical algebra.
pub struct TropicalAlgebra<F: Semiring, const N: usize> {
  bilinear_form: BilinearForm<F, N>,
}

impl<F: Semiring + Copy, const N: usize> TropicalAlgebra<F, N>
where [(); 1 << N]:
{
  /// Creates a new tropical algebra with the given bilinear form.
  ///
  /// # Arguments
  ///
  /// * `bilinear_form` - The bilinear form defining the algebra
  ///
  /// # Examples
  ///
  /// ```
  /// #![feature(generic_const_exprs)]
  /// use harness_algebra::{
  ///   semimodule::tropical::{BilinearForm, TropicalAlgebra},
  ///   vector::Vector,
  /// };
  ///
  /// let bilinear_form = BilinearForm::new(Vector::<3, f64>([1.0, 1.0, -1.0]));
  /// let algebra = TropicalAlgebra::new(bilinear_form);
  /// ```
  pub const fn new(bilinear_form: BilinearForm<F, N>) -> Self { Self { bilinear_form } }

  /// Creates a new element in the algebra from a vector of coefficients.
  ///
  /// # Arguments
  ///
  /// * `value` - A vector of coefficients for each basis blade
  ///
  /// # Examples
  ///
  /// ```
  /// #![feature(generic_const_exprs)]
  /// use harness_algebra::{
  ///   semimodule::tropical::{QuadraticForm, TropicalAlgebra},
  ///   vector::Vector,
  /// };
  ///
  /// let algebra = TropicalAlgebra::new(QuadraticForm::new(Vector::<3, f64>([1.0, 1.0, -1.0])));
  /// let element = algebra.element(Vector::<8, f64>([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
  /// ```
  pub const fn element(&self, value: Vector<{ 1 << N }, F>) -> TropicalElement {
    TropicalElement { value, bilinear_form: Some(&self.bilinear_form) }
  }
}
