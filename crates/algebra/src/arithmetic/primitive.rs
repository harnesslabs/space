//! # Primitive Types Implementation
//!
//! This module implements the algebraic traits from the crate for Rust's primitive numeric types.
//!
//! ## Implementations
//!
//! - `Additive`: Implemented for all primitive numeric types, indicating they form an additive
//!   structure
//! - `Multiplicative`: Implemented for all primitive numeric types, indicating they form a
//!   multiplicative structure
//! - `Infinity`: Implemented for floating point types (f32, f64), providing access to their
//!   infinity values
//!
//! These trait implementations allow primitive types to be used directly with the algebraic
//! abstractions defined in this crate, enabling seamless integration between Rust's built-in
//! types and the algebraic structures defined in this library.
//!
//! No additional methods are needed as Rust's primitive types already implement the required
//! operations (`Add`, `Mul`, etc.) with the correct semantics.

use super::*;

// Implement Additive for all primitive numeric types
impl Additive for u8 {}
impl Additive for u16 {}
impl Additive for u32 {}
impl Additive for u64 {}
impl Additive for u128 {}
impl Additive for usize {}

impl Additive for i8 {}
impl Additive for i16 {}
impl Additive for i32 {}
impl Additive for i64 {}
impl Additive for i128 {}
impl Additive for isize {}

impl Additive for f32 {}
impl Additive for f64 {}

// Implement Multiplicative for all primitive numeric types
impl Multiplicative for u8 {}
impl Multiplicative for u16 {}
impl Multiplicative for u32 {}
impl Multiplicative for u64 {}
impl Multiplicative for u128 {}
impl Multiplicative for usize {}

impl Multiplicative for i8 {}
impl Multiplicative for i16 {}
impl Multiplicative for i32 {}
impl Multiplicative for i64 {}
impl Multiplicative for i128 {}
impl Multiplicative for isize {}

impl Multiplicative for f32 {}
impl Multiplicative for f64 {}

// Implement Infinity for float primitive numeric types
impl Infinity for f32 {
  const INFINITY: Self = Self::INFINITY;
}

impl Infinity for f64 {
  const INFINITY: Self = Self::INFINITY;
}
