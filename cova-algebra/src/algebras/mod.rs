//! Implementation of an [`Algebra`] interface and various algebras
//!
//! This module provides implementations of different types of algebras, which are algebraic
//! structures that combine the properties of vector spaces with multiplication operations. An
//! algebra is a vector space equipped with a bilinear product that satisfies certain properties.
//!
//!
//! # Key Concepts
//!
//! - **Algebra**: A vector space equipped with a bilinear product that is compatible with the
//!   vector space operations. The product must satisfy the distributive laws with respect to
//!   addition and scalar multiplication.
//!
//! - **Clifford Algebra**: A type of algebra that generalizes the real numbers, complex numbers,
//!   and quaternions. It is particularly useful in geometry and physics for representing rotations,
//!   reflections, and other transformations.
//!
//! # Implementations
//!
//! Currently, this module provides:
//!
//! - [`clifford`]: Implementation of Clifford algebras, which are useful for geometric computations
//!   and transformations in n-dimensional spaces.
//! - [`boolean`]: Implementation of Boolean algebra, which is useful for logical operations and
//!   boolean logic.

use super::*;
use crate::{
  arithmetic::Multiplicative,
  modules::{TwoSidedModule, VectorSpace},
  rings::Field,
};

pub mod boolean;
pub mod clifford;

/// Trait defining the requirements for an algebra.
///
/// An algebra is a vector space equipped with a bilinear product that satisfies:
/// - Distributivity: a(b + c) = ab + ac and (a + b)c = ac + bc
/// - Compatibility with scalar multiplication: (ka)b = k(ab) = a(kb)
///
/// This trait combines the properties of a vector space with those of a multiplicative structure,
/// ensuring that the algebra's operations are compatible with both the vector space and ring
/// operations.
pub trait Algebra: VectorSpace + Multiplicative
where <Self as TwoSidedModule>::Ring: Field {
}
