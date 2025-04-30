use core::marker::PhantomData;

use crate::{
  arithmetic::{Add, AddAssign, Additive, Mul, Neg, Sub, SubAssign, Zero},
  group::{AbelianGroup, Group},
  ring::{Field, Ring},
};

pub trait Module: AbelianGroup + Mul<Self::Ring, Output = Self> {
  type Ring: Ring;
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct TrivialModule<R> {
  pub(crate) _r: PhantomData<R>,
}

impl<R: Ring> Module for TrivialModule<R> {
  type Ring = R;
}

impl<R> Add for TrivialModule<R> {
  type Output = Self;

  fn add(self, _: Self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> AddAssign for TrivialModule<R> {
  fn add_assign(&mut self, _: Self) {}
}

impl<R> Sub for TrivialModule<R> {
  type Output = Self;

  fn sub(self, _: Self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> SubAssign for TrivialModule<R> {
  fn sub_assign(&mut self, _: Self) {}
}

impl<R> Neg for TrivialModule<R> {
  type Output = Self;

  fn neg(self) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> Mul<R> for TrivialModule<R> {
  type Output = Self;

  fn mul(self, _: R) -> Self::Output { Self { _r: PhantomData } }
}

impl<R> Zero for TrivialModule<R> {
  fn zero() -> Self { Self { _r: PhantomData } }

  fn is_zero(&self) -> bool { true }
}

impl<R: Ring> Additive for TrivialModule<R> {}

impl<R> Group for TrivialModule<R> {
  fn identity() -> Self { Self { _r: PhantomData } }

  fn inverse(&self) -> Self { Self { _r: PhantomData } }
}

impl<R: Ring> AbelianGroup for TrivialModule<R> {}
