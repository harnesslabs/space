//! Implementation of various algebraic structures.
//!
//! This module provides implementations of different types of algebras, which are algebraic
//! structures that combine the properties of vector spaces with multiplication operations. An
//! algebra is a vector space equipped with a bilinear product that satisfies certain properties.
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
//! - **Tropical Algebra**: A semiring where addition is replaced by minimum (or maximum) and
//!   multiplication is replaced by addition. Useful in optimization, scheduling, and path finding.
//!
//! # Implementations
//!
//! Currently, this module provides:
//!
//! - [`clifford`]: Implementation of Clifford algebras, which are useful for geometric computations
//!   and transformations in n-dimensional spaces.
//! - [`tropical`]: Implementation of tropical algebras, which are useful for optimization and
//!   scheduling problems.

use crate::{arithmetic::Multiplicative, module::TwoSidedModule, ring::Field, vector::VectorSpace};

pub mod clifford;
pub mod tropical;

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
