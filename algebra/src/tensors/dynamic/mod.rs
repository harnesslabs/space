//! # Dynamic Tensors Module
//!
//! This module provides implementations of tensors with dynamically determined dimensions,
//! focusing on vectors and matrices.
//!
//! ## Overview
//!
//! The dynamic tensors module includes:
//!
//! - `DynamicVector`: A flexible vector implementation with arbitrary dimension
//! - `DynamicDenseMatrix`: A matrix implementation with two storage orientation options:
//!   - Row-major: Efficient for row operations
//!   - Column-major: Efficient for column operations
//!
//! ## Mathematical Foundation
//!
//! Tensors are mathematical objects that generalize vectors and matrices to higher dimensions.
//! The implementations in this module adhere to the algebraic properties of vector spaces
//! and linear transformations over arbitrary fields.
//!
//! ## Performance Considerations
//!
//! - Choose storage orientation based on the dominant operation pattern:
//!   - Use row-major matrices when mostly operating on rows
//!   - Use column-major matrices when mostly operating on columns
//! - Matrix-vector operations automatically use the most efficient implementation based on the
//!   storage orientation
//!
//! ## Examples
//!
//! ```
//! use harness_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{
//!     matrix::{DynamicDenseMatrix, RowMajor},
//!     vector::DynamicVector,
//!   },
//! };
//!
//! // Create vectors
//! let v1 = DynamicVector::from([1.0, 2.0, 3.0]);
//! let v2 = DynamicVector::from([4.0, 5.0, 6.0]);
//!
//! // Create a matrix from rows
//! let mut matrix = DynamicDenseMatrix::<f64, RowMajor>::new();
//! matrix.append_row(v1);
//! matrix.append_row(v2);
//!
//! // Perform Gaussian elimination to row echelon form
//! let result = matrix.row_echelon_form();
//! ```

use super::*;

pub mod matrix;
pub mod vector;
