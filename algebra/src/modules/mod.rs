// TODO: Redo these docs.

//! Module theory abstractions and implementations.
//!
//! This module provides traits and implementations for module theory concepts,
//! which generalize vector spaces by allowing the scalars to lie in a ring rather than a field.
//!
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

use core::marker::PhantomData;

use super::*;
use crate::{
  groups::{AbelianGroup, Group},
  rings::{Ring, Semiring},
};

mod trivial;
mod tropical;

/// A left semimodule over a semiring.
///
/// A set with commutative addition and left scalar multiplication satisfying:
/// - Distributivity: s * (x + y) = s * x + s * y
/// - Compatibility: (s + t) * x = s * x + t * x
/// - Associativity: (s * t) * x = s * (t * x)
/// - Identity: 1 * x = x
/// - Zero: 0 * x = 0
pub trait LeftSemimodule
where Self: Mul<Self::Semiring> {
  /// The Semiring that this semimodule is defined over.
  type Semiring: Semiring;
}

/// A right semimodule over a semiring.
///
/// A set with commutative addition and right scalar multiplication satisfying:
/// - Distributivity: (x + y) * s = x * s + y * s
/// - Compatibility: x * (s + t) = x * s + x * t
/// - Associativity: x * (s * t) = (x * s) * t
/// - Identity: x * 1 = x
/// - Zero: x * 0 = 0
pub trait RightSemimodule
where Self: Mul<Self::Semiring> {
  /// The Semiring that this semimodule is defined over.
  type Semiring: Semiring;
}

/// A two-sided semimodule over a semiring.
///
/// - **Semimodule**: A vector space generalization where scalars are elements of a semiring
/// - **Left/Right Semimodule**: Defines scalar multiplication from left/right respectively
/// - **Two-Sided Semimodule**: Both left and right semimodule over the same semiring
///
/// Combines left and right semimodule properties over the same semiring.
/// Note: For commutative semirings, left and right actions typically coincide.
pub trait TwoSidedSemimodule: LeftSemimodule + RightSemimodule {
  /// The semiring over which this semimodule is defined.
  type Semiring: Semiring;
}

/// A trait representing a left module over a ring.
///
/// A left module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on the left.
pub trait LeftModule: AbelianGroup
where Self::Ring: Mul<Self> {
  /// The ring over which this module is defined.
  type Ring: Ring;
}

/// A trait representing a right module over a ring.
///
/// A right module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on the right.
pub trait RightModule: AbelianGroup
where Self::Ring: Mul<Self> {
  /// The ring over which this module is defined.
  type Ring: Ring;
}

/// A trait representing a two-sided module over a ring.
///
/// A two-sided module is a generalization of a vector space, where the scalars lie in a ring
/// rather than a field. This trait combines the requirements for an Abelian group
/// with scalar multiplication by elements of the ring on both the left and right.
pub trait TwoSidedModule: LeftModule + RightModule {
  /// The ring over which this module is defined.
  type Ring: Ring;
}
