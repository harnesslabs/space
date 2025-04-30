use crate::{arithmetic::Multiplicative, module::TwoSidedModule, ring::Field, vector::VectorSpace};

pub mod clifford;

pub trait Algebra: VectorSpace + Multiplicative
where <Self as TwoSidedModule>::Ring: Field {
}
