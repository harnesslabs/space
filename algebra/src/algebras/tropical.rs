//! Implementation of tropical algebra.
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
//! - Path finding
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
    arithmetic::{Add, AddAssign, Mul, MulAssign, One, Zero, Additive, Multiplicative},
    semiring::Semiring,
};

/// A tropical algebra element.
///
/// This represents a value in the tropical semiring (ℝ ∪ {-∞}, max, +).
/// The value is stored as an `f64`, with `f64::NEG_INFINITY` representing -∞.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TropicalElement {
    value: f64,
}

impl TropicalElement {
    /// Creates a new tropical element with the given value.
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    /// Returns the value of this tropical element.
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Display for TropicalElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Add for TropicalElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { value: self.value.max(rhs.value) }
    }
}

impl AddAssign for TropicalElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for TropicalElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self { value: self.value + rhs.value }
    }
}

impl MulAssign for TropicalElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Zero for TropicalElement {
    fn zero() -> Self {
        Self { value: f64::NEG_INFINITY }
    }

    fn is_zero(&self) -> bool {
        self.value == f64::NEG_INFINITY
    }
}

impl One for TropicalElement {
    fn one() -> Self {
        Self { value: 0.0 }
    }
}

impl Additive for TropicalElement {}
impl Multiplicative for TropicalElement {}

impl Semiring for TropicalElement {
    fn zero() -> Self {
        <Self as Zero>::zero()
    }

    fn one() -> Self {
        <Self as One>::one()
    }
}

/// A tropical algebra.
///
/// This struct provides a way to create tropical elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TropicalAlgebra;

impl TropicalAlgebra {
    /// Creates a new tropical algebra.
    pub fn new() -> Self {
        Self
    }

    /// Creates a new tropical element with the given value.
    pub fn element(&self, value: f64) -> TropicalElement {
        TropicalElement::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let algebra = TropicalAlgebra::new();
        let a = algebra.element(3.0);
        let b = algebra.element(5.0);
        let sum = a + b;
        // In max-plus tropical algebra, addition is max
        assert_eq!(sum.value(), 5.0);
    }

    #[test]
    fn test_multiplication() {
        let algebra = TropicalAlgebra::new();
        let a = algebra.element(3.0);
        let b = algebra.element(5.0);
        let product = a * b;
        // In tropical algebra, multiplication is addition
        assert_eq!(product.value(), 8.0);
    }

    #[test]
    fn test_zero() {
        let algebra = TropicalAlgebra::new();
        let zero = <TropicalElement as Zero>::zero();
        let a = algebra.element(3.0);
        let sum = a + zero;
        // Adding zero (-∞) should return the other element
        assert_eq!(sum.value(), 3.0);
    }

    #[test]
    fn test_one() {
        let algebra = TropicalAlgebra::new();
        let one = <TropicalElement as One>::one();
        let a = algebra.element(3.0);
        let product = a * one;
        // Multiplying by one (0) should return the other element
        assert_eq!(product.value(), 3.0);
    }
}
