use crate::arithmetic::{Additive, Multiplicative};

use super::*;

pub trait Group {
  fn identity() -> Self;
  fn inverse(&self) -> Self;
}

/// Group trait defined by a binary operation, identity element and inverse.
pub trait AbelianGroup: Group + Additive {}

pub trait NonAbelianGroup: Group + Multiplicative {}
