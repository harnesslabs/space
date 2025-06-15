//! # Tensors Module
//!
//! This module provides implementations of tensors - mathematical objects that
//! generalize vectors and matrices to higher-order dimensions.
//!
//! ## Mathematical Background
//!
//! Tensors are multi-dimensional arrays that transform according to specific rules
//! under changes of coordinates. In this library, they are implemented as concrete
//! representations of vector spaces over arbitrary fields.
//!
//! ## Module Organization
//!
//! The tensors module is organized into two primary submodules:
//!
//! - [`fixed`]: Implementations of tensors with dimensions known at compile-time using const
//!   generics for improved performance and type safety.
//!
//! - [`dynamic`]: Implementations of tensors with dimensions determined at runtime, offering
//!   greater flexibility for applications where tensor sizes vary.
//!
//! ## Algebraic Structure
//!
//! All tensor implementations satisfy the algebraic properties of vector spaces:
//!
//! - They form abelian groups under addition
//! - They support scalar multiplication with elements from a field
//! - They implement the vector space axioms (distributivity, associativity, etc.)
//!
//! ## Example Usage
//!
//! ```
//! use cova_algebra::tensors::{dynamic::Vector, fixed::FixedVector};
//!
//! // Fixed-size vector (dimension known at compile time)
//! let fixed = FixedVector::<3, f64>([1.0, 2.0, 3.0]);
//!
//! // Dynamic vector (dimension determined at runtime)
//! let dynamic = Vector::<f64>::from([4.0, 5.0, 6.0]);
//! ```

pub use nalgebra::*;

use super::*;
use crate::{
  groups::{AbelianGroup, Group},
  modules::{LeftModule, RightModule, TwoSidedModule, VectorSpace},
  rings::Field,
};
pub mod fixed;


use nalgebra::Scalar;