//! Semimodule abstractions and implementations.
//!
//! A semimodule is a generalization of vector spaces and modules where the scalars form a semiring
//! rather than a ring or field. Like modules, semimodules support scalar multiplication and
//! addition, but the scalars only need to form a semiring structure.
//!
//! # Key Concepts
//!
//! - **Semimodule**: A set with an additive structure and scalar multiplication by elements of a
//!   semiring
//! - **Semiring**: A ring-like structure but without the requirement of additive inverses
//! - **Scalar Multiplication**: Compatible multiplication between semiring elements and semimodule
//!   elements
//!
//! # Examples
//!
//! The tropical algebra (implemented in the `tropical` module) is a classic example of a
//! semimodule, where addition is replaced by maximum and multiplication by regular addition.
//!
//! # Module Structure
//!
//! - [`Semimodule`]: Main trait for semimodule structures
//! - [`tropical`]: Implementation of tropical algebra as a semimodule

use crate::{
  arithmetic::Multiplicative, // vector::VectorSpace,
  module::TwoSidedSemimodule,
  ring::Semiring,
};

/// A semimodule is a generalization of vector spaces and modules where the scalars form a
/// semiring rather than a ring or field. Like modules, semimodules support scalar multiplication
/// and addition, but the scalars only need to form a semiring structure.
pub trait Semimodule: Multiplicative + TwoSidedSemimodule
where <Self as TwoSidedSemimodule>::Semiring: Semiring {
}
pub mod tropical;
