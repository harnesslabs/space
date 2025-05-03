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
//! use harness_algebra::{group::Group, modular, ring::Ring};
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
//! use harness_algebra::{
//!   ring::Field,
//!   vector::{Vector, VectorSpace},
//! };
//!
//! let v1 = Vector::<3, f64>([1.0, 2.0, 3.0]);
//! let v2 = Vector::<3, f64>([4.0, 5.0, 6.0]);
//! let sum = v1 + v2;
//! ```
//!
//! # Modules
//!
//! - [`arithmetic`]: Basic arithmetic traits and operations
//! - [`group`]: Group theory abstractions and implementations
//! - [`ring`]: Ring theory abstractions and implementations
//! - [`module`]: Module theory abstractions and implementations
//! - [`vector`]: Vector space abstractions and implementations
//! - [`modular`]: Modular arithmetic abstractions and implementations

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod algebras;
pub mod arithmetic;
pub mod group;
pub mod modular;
pub mod module;
pub mod ring;
pub mod semiring;
pub mod vector;
