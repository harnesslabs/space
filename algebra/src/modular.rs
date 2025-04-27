#[macro_export]
macro_rules! modular {
  ($name:ident, $inner:ty, $modulus:expr) => {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct $name($inner);

    impl $name {
      pub const MODULUS: $inner = $modulus;

      pub fn new(value: $inner) -> Self {
        Self(value % Self::MODULUS)
      }

      pub fn value(&self) -> $inner {
        self.0
      }
    }

    impl num::Zero for $name {
      fn zero() -> Self {
        Self(0)
      }

      fn is_zero(&self) -> bool {
        self.0 == 0
      }
    }

    impl std::ops::Add for $name {
      type Output = Self;
      fn add(self, rhs: Self) -> Self {
        Self::new(self.0.wrapping_add(rhs.0))
      }
    }

    impl std::ops::AddAssign for $name {
      fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
      }
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
      fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
      }
    }

    impl std::ops::SubAssign for $name {
      fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
      }
    }

    impl num::Bounded for $name {
      fn min_value() -> Self {
        Self(0)
      }

      fn max_value() -> Self {
        Self(Self::MODULUS - 1)
      }
    }

    impl num::One for $name {
      fn one() -> Self {
        Self(1)
      }
    }

    impl std::ops::Mul for $name {
      type Output = Self;
      fn mul(self, rhs: Self) -> Self {
        Self::new(self.0.wrapping_mul(rhs.0))
      }
    }

    impl std::ops::MulAssign for $name {
      fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
      }
    }

    impl $crate::arithmetic::Additive for $name {}
    impl $crate::arithmetic::Multiplicative for $name {}

    impl $crate::group::Group for $name {
      fn identity() -> Self {
        Self(0)
      }

      fn inverse(&self) -> Self {
        Self(Self::MODULUS - self.0)
      }
    }

    impl $crate::group::AbelianGroup for $name {}
    impl $crate::ring::Ring for $name {
      fn one() -> Self {
        Self(1)
      }

      fn zero() -> Self {
        Self(0)
      }
    }
  };
}

#[cfg(test)]
mod tests {
  use crate::group::Group;
  use crate::ring::Ring;

  modular!(Mod7, u32, 7);

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
}
