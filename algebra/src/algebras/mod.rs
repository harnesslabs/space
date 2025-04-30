use crate::{arithmetic::Multiplicative, ring::Field, vector::VectorSpace};

pub mod clifford;

pub trait Algebra: VectorSpace + Multiplicative
where Self::Ring: Field {
}
