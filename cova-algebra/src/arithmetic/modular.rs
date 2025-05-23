//! Modular arithmetic abstractions and implementations.
//!
//! This module provides a macro for creating custom modular number types
//! and implementations of various arithmetic operations for them.
//!
//! # Examples
//!
//! ```
//! use cova_algebra::{modular, prime_field};
//!
//! // Create a type for numbers modulo 7 (a prime number)
//! modular!(Mod7, u32, 7);
//! prime_field!(Mod7);
//!
//! let a = Mod7::new(3);
//! let b = Mod7::new(5);
//! let sum = a + b; // 8 ≡ 1 (mod 7)
//!
//! let product = a * b; // 15 ≡ 1 (mod 7)
//!
//! let inverse = a.multiplicative_inverse(); // 3 * 5 ≡ 1 (mod 7)
//! ```

/// A const function to check if a number is prime at compile time.
///
/// This is used by the `modular!` macro to determine if a field implementation
/// should be generated.
pub const fn is_prime(n: u32) -> bool {
  if n <= 1 {
    return false;
  }
  if n <= 3 {
    return true;
  }
  if n % 2 == 0 || n % 3 == 0 {
    return false;
  }
  let mut i = 5;
  while i * i <= n {
    if n % i == 0 || n % (i + 2) == 0 {
      return false;
    }
    i += 6;
  }
  true
}

/// A macro for creating custom modular number types.
///
/// This macro creates a new type for numbers modulo a given value,
/// implementing various arithmetic operations and algebraic traits.
/// If the modulus is a prime number, it will also implement the `Field` trait.
///
/// # Examples
///
/// ```
/// use cova_algebra::modular;
///
/// // Create a type for numbers modulo 7 (a prime number)
/// modular!(Mod7, u32, 7);
///
/// let a = Mod7::new(3);
/// let b = Mod7::new(5);
///
/// // Addition: 3 + 5 = 8 ≡ 1 (mod 7)
/// let sum = a + b;
/// assert_eq!(sum.value(), 1);
///
/// // Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
/// let product = a * b;
/// assert_eq!(product.value(), 1);
/// ```
#[macro_export]
macro_rules! modular {
  ($name:ident, $inner:ty, $modulus:expr) => {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct $name($inner);

    impl $name {
      /// The modulus for this modular number type.
      pub const MODULUS: $inner = $modulus;

      /// Creates a new modular number from a value.
      ///
      /// The value is automatically reduced modulo `MODULUS`.
      pub fn new(value: $inner) -> Self { Self(value % Self::MODULUS) }

      /// Returns the value of this modular number.
      #[allow(unused)]
      pub const fn value(&self) -> $inner { self.0 }
    }

    impl num_traits::Zero for $name {
      fn zero() -> Self { Self(0) }

      fn is_zero(&self) -> bool { self.0 == 0 }
    }

    impl std::ops::Add for $name {
      type Output = Self;

      fn add(self, rhs: Self) -> Self { Self::new(self.0.wrapping_add(rhs.0)) }
    }

    impl std::ops::AddAssign for $name {
      fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
    }

    impl std::ops::Neg for $name {
      type Output = Self;

      fn neg(self) -> Self {
        if self.0 == 0 {
          self
        } else {
          Self::new(Self::MODULUS - self.0)
        }
      }
    }

    impl std::ops::Sub for $name {
      type Output = Self;

      fn sub(self, rhs: Self) -> Self { self + (-rhs) }
    }

    impl std::ops::SubAssign for $name {
      fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
    }

    impl num_traits::Bounded for $name {
      fn min_value() -> Self { Self(0) }

      fn max_value() -> Self { Self(Self::MODULUS - 1) }
    }

    impl num_traits::One for $name {
      fn one() -> Self { Self(1) }
    }

    impl std::ops::Mul for $name {
      type Output = Self;

      fn mul(self, rhs: Self) -> Self { Self::new(self.0.wrapping_mul(rhs.0)) }
    }

    impl std::ops::MulAssign for $name {
      fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
    }

    impl $crate::arithmetic::Additive for $name {}
    impl $crate::arithmetic::Multiplicative for $name {}

    impl $crate::groups::Group for $name {
      fn identity() -> Self { Self(0) }

      fn inverse(&self) -> Self { Self(Self::MODULUS - self.0) }
    }

    impl $crate::groups::AbelianGroup for $name {}
    impl $crate::rings::Ring for $name {}

    impl From<$inner> for $name {
      fn from(value: $inner) -> Self { Self::new(value) }
    }
  };
}

/// A macro for creating prime field types.
///
/// This macro extends the given type with a method for computing the modular
/// multiplicative inverse using Fermat's Little Theorem.
///
/// # Examples
///
/// ```
/// use cova_algebra::{modular, prime_field};
///
/// modular!(Mod7, u32, 7);
/// prime_field!(Mod7);
///
/// let a = Mod7::new(3);
/// let inverse = a.multiplicative_inverse();
/// assert_eq!(inverse.value(), 5); // 3 * 5 ≡ 1 (mod 7)
/// ```
#[macro_export]
macro_rules! prime_field {
  ($inner:ty) => {
    impl $inner {
      /// Computes the modular multiplicative inverse using Fermat's Little Theorem.
      ///
      /// # Panics
      ///
      /// This function will panic if called on zero.
      fn multiplicative_inverse(&self) -> Self {
        if self.0 == 0 {
          panic!("Cannot compute inverse of zero");
        }
        // Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p)
        // Therefore, a^(p-2) ≡ a^(-1) (mod p)
        let mut result = Self(1);
        let mut base = *self;
        let mut exponent = Self::MODULUS - 2;
        while exponent > 0 {
          if exponent % 2 == 1 {
            result *= base;
          }
          base = base * base;
          exponent /= 2;
        }
        result
      }
    }

    impl std::ops::Div for $inner
    where [(); $crate::arithmetic::modular::is_prime(<$inner>::MODULUS) as usize - 1]:
    {
      type Output = Self;

      #[allow(clippy::suspicious_arithmetic_impl)]
      fn div(self, rhs: Self) -> Self { self * rhs.multiplicative_inverse() }
    }

    impl std::ops::DivAssign for $inner
    where [(); $crate::arithmetic::modular::is_prime(<$inner>::MODULUS) as usize - 1]:
    {
      fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
    }

    impl $crate::rings::Field for $inner
    where [(); $crate::arithmetic::modular::is_prime(<$inner>::MODULUS) as usize - 1]:
    {
      fn multiplicative_inverse(&self) -> Self { self.multiplicative_inverse() }
    }
  };
}

#[cfg(test)]
mod tests {
  use crate::{
    arithmetic::{One, Zero},
    groups::Group,
  };

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);

  #[test]
  fn test_modular_group() {
    // Test modulo 7 arithmetic
    let a = Mod7::new(3); // 3 mod 7
    let b = Mod7::new(5); // 5 mod 7

    // Test addition: 3 + 5 = 8 ≡ 1 (mod 7)
    let sum = a + b;
    assert_eq!(sum.value(), 1);

    // Test subtraction: 3 - 5 = -2 ≡ 5 (mod 7)
    let diff = a - b;
    assert_eq!(diff.value(), 5);

    // Test negation: -3 ≡ 4 (mod 7) and inverse
    let neg = -a;
    assert_eq!(neg.value(), 4);
    assert_eq!(neg.inverse().value(), 3);

    // Test zero and identity
    assert_eq!(Mod7::zero().value(), 0);
    assert_eq!(Mod7::identity().value(), 0);
  }

  #[test]
  fn test_modular_ring() {
    let a = Mod7::new(3);
    let b = Mod7::new(5);

    let sum = a * b;
    assert_eq!(sum.value(), 1);

    let product = a * a;
    assert_eq!(product.value(), 2);

    // Test additive inverse
    let inverse = a.inverse();
    assert_eq!(inverse.value(), 4);

    // Test one
    assert_eq!(Mod7::one().value(), 1);

    // Test zero
    assert_eq!(Mod7::zero().value(), 0);
  }

  #[test]
  fn test_modular_field() {
    let a = Mod7::new(3);
    let inverse = a.multiplicative_inverse();
    assert_eq!(inverse.value(), 5); // 3 * 5 ≡ 1 (mod 7)
  }
}
