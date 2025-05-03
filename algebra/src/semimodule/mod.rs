
use crate::{
  module::TwoSidedSemimodule,
  ring::Semiring,
  vector::VectorSpace,
};


pub trait Semimodule: VectorSpace + Multiplicative
where <Self as TwoSidedSemimodule>::Semiring: Semiring {
}
pub mod tropical;