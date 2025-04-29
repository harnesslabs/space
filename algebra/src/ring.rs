use crate::{arithmetic::Multiplicative, group::AbelianGroup};

use super::*;

pub trait Ring: AbelianGroup + Multiplicative {
  fn one() -> Self;
  fn zero() -> Self;
}

pub trait Field: Ring + Div + DivAssign {
  fn multiplicative_inverse(&self) -> Self;
}
