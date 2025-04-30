use super::*;
use crate::arithmetic::{Additive, Multiplicative};

pub trait Group {
  fn identity() -> Self;
  fn inverse(&self) -> Self;
}

/// Group trait defined by a binary operation, identity element and inverse.
pub trait AbelianGroup:
  Group + Zero + Additive + Neg<Output = Self> + Sub<Output = Self> + SubAssign
{
}

pub trait NonAbelianGroup: Group + One + Multiplicative + Div<Output = Self> + DivAssign {}
