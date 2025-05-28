//! # Cova Solvers
//!
//! A collection of convex optimization solvers including:
//! - Linear Programming (Simplex method)
//! - ADMM (Alternating Direction Method of Multipliers)
//!
//! ## Example
//!
//! ```rust
//! use cova_algebra::tensors::{DMatrix, DVector};
//! use cova_solver::linear_programming::SimplexSolver;
//!
//! // Minimize c^T x subject to Ax <= b, x >= 0
//! let c = DVector::from_vec(vec![-1.0, -2.0]); // coefficients to minimize
//! let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 1.0]);
//! let b = DVector::from_vec(vec![3.0, 4.0]);
//!
//! let mut solver = SimplexSolver::new();
//! let result = solver.solve(&c, &a, &b);
//! ```

pub mod admm;
pub mod error;
pub mod linear_programming;
pub mod traits;

use cova_algebra::tensors::DMatrix;
use error::{SolverError, SolverResult};
pub use traits::Solver;

pub mod prelude {
  pub use crate::{error::SolverError, traits::Solver};
}
