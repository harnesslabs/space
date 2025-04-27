use super::*;
use std::ops::Add;

use num::Bounded;

use crate::arithmetic::Additive;
use crate::group::{AbelianGroup, Group};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Modular<T: Additive + Bounded + PartialEq>(T);

impl<T: Additive + Bounded> Modular<T> {
  pub fn new(value: T) -> Self {
    Self(value)
  }
}

impl<T: Additive + Bounded> Group for Modular<T> {
  fn identity() -> Self {
    Self(T::zero())
  }

  fn inverse(&self) -> Self {
    Self(T::max_value() - self.0)
  }
}
impl<T: Additive + Bounded + PartialEq> Additive for Modular<T> {}
impl<T: Additive + Bounded + PartialEq> AbelianGroup for Modular<T> {}

impl<T: Additive + Bounded> Add for Modular<T> {
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output {
    Self(self.0 + rhs.0)
  }
}

impl<T: Additive + Bounded> AddAssign for Modular<T> {
  fn add_assign(&mut self, rhs: Self) {
    self.0 += rhs.0;
  }
}

impl<T: Additive + Bounded> Sub for Modular<T> {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output {
    Self(self.0 - rhs.0)
  }
}

impl<T: Additive + Bounded> SubAssign for Modular<T> {
  fn sub_assign(&mut self, rhs: Self) {
    self.0 -= rhs.0;
  }
}

impl<T: Additive + Bounded> Neg for Modular<T> {
  type Output = Self;

  fn neg(self) -> Self::Output {
    Self(-self.0)
  }
}

impl<T: Additive + Bounded> Zero for Modular<T> {
  fn zero() -> Self {
    Self(T::zero())
  }

  fn is_zero(&self) -> bool {
    self.0 == T::zero()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_modular() {}
}
