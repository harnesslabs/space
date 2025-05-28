//! # Cova Solvers
//!
//! A collection of convex optimization solvers including:
//! - Linear Programming (Simplex method)
//! - ADMM (Alternating Direction Method of Multipliers)
//!
//! ## Example
//!
//! ```rust
//! use cova_algebra::{
//!   prelude::*,
//!   tensors::dynamic::{Matrix, Vector},
//! };
//! use cova_solver::linear_programming::SimplexSolver;
//!
//! // Minimize c^T x subject to Ax <= b, x >= 0
//! let c = Vector::from_vec(vec![-1.0, -2.0]); // coefficients to minimize
//! let a = Matrix::builder().row([1.0, 1.0]).row([2.0, 1.0]).build();
//! let b = Vector::from_vec(vec![3.0, 4.0]);
//!
//! let mut solver = SimplexSolver::new();
//! let result = solver.solve(&c, &a, &b);
//! ```

pub mod admm;
pub mod error;
pub mod linear_programming;
pub mod traits;

use cova_algebra::tensors::dynamic::{Matrix, Vector};
use error::{SolverError, SolverResult};
pub use traits::Solver;

pub mod prelude {
  pub use crate::{error::SolverError, traits::Solver};
}
