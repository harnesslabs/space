//! A Rust library for abstract algebra.
//!
//! This crate provides implementations of various algebraic structures and operations,
//! with a focus on modular arithmetic and abstract algebra concepts.
//!
//! # Features
//!
//! - **Modular Arithmetic**: Create custom modular number types with the `modular!` macro
//! - **Abstract Algebra**: Implementations of fundamental algebraic structures:
//!   - Groups (both Abelian and Non-Abelian)
//!   - Rings
//!   - Fields
//!   - Modules
//!   - Vector Spaces
//!
//! # Examples
//!
//! ## Modular Arithmetic
//!
//! ```
//! use cova_algebra::{algebras::boolean::Boolean, modular, rings::Field};
//!
//! // Create a type for numbers modulo 7
//! modular!(Mod7, u32, 7);
//!
//! let a = Mod7::new(3);
//! let b = Mod7::new(5);
//! let sum = a + b; // 8 ≡ 1 (mod 7)
//! ```
//!
//! ## Vector Spaces
//!
//! ```
//! use cova_algebra::{algebras::boolean::Boolean, rings::Field, tensors::SVector};
//!
//! let v1 = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
//! let v2 = SVector::<f64, 3>::from_row_slice(&[4.0, 5.0, 6.0]);
//! let sum = v1 + v2;
//! ```

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod algebras;
pub mod arithmetic;
pub mod category;
pub mod groups;
pub mod modules;
pub mod rings;
pub mod tensors;

pub use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use num_traits::{One, Zero};

pub use crate::arithmetic::{Additive, Multiplicative};

pub mod prelude {
  //! # Prelude Module
  //!
  //! This module re-exports the most commonly used types, traits, and operations from the
  //! algebra crate for convenient importing.
  //!
  //! ## Purpose
  //!
  //! The prelude pattern allows users to import multiple commonly used items with a single
  //! import statement, reducing boilerplate and improving code readability.
  //!
  //! ## Contents
  //!
  //! The prelude includes:
  //!
  //! - Core algebraic structures: [`Algebra`], [`Group`], [`Ring`], [`Field`], [`VectorSpace`]
  //! - Behavioral traits: [`Additive`], [`Multiplicative`]
  //! - Group variants: [`AbelianGroup`], [`NonAbelianGroup`]
  //! - Module types: [`LeftModule`], [`RightModule`], [`TwoSidedModule`]
  //! - Semimodule types: [`LeftSemimodule`], [`RightSemimodule`], [`TwoSidedSemimodule`]
  //! - Fundamental operators: [`Add`], [`Mul`], [`Sub`], [`Div`] and their assignment variants
  //! - Identity concepts: [`Zero`], [`One`], [`Neg`]
  //!
  //! ## Usage
  //!
  //! ```
  //! // Import everything from the prelude
  //! use cova_algebra::prelude::*;
  //! ```

  pub use crate::{
    algebras::Algebra,
    arithmetic::{Additive, Multiplicative},
    category::Category,
    groups::{AbelianGroup, Group, NonAbelianGroup},
    modules::{
      LeftModule, LeftSemimodule, RightModule, RightSemimodule, TwoSidedModule, TwoSidedSemimodule,
      VectorSpace,
    },
    rings::{Field, Ring},
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, One, Sub, SubAssign, Zero,
  };
}

#[cfg(test)]
mod fixtures {
  use crate::{modular, prime_field};

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);
}
