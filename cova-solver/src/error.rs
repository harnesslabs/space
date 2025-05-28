//! Error types for the cova-solvers library

use thiserror::Error;

pub type SolverResult<T> = Result<T, SolverError>;

/// Errors that can occur during optimization solving
#[derive(Error, Debug, Clone)]
pub enum SolverError {
  /// The problem is infeasible (no solution exists)
  #[error("Problem is infeasible")]
  Infeasible,

  /// The problem is unbounded (objective can be made arbitrarily good)
  #[error("Problem is unbounded")]
  Unbounded,

  /// The solver failed to converge within the maximum iterations
  #[error("Solver failed to converge within {max_iterations} iterations")]
  ConvergenceFailure { max_iterations: usize },

  /// Numerical issues encountered during solving
  #[error("Numerical error: {message}")]
  NumericalError { message: String },

  /// Invalid problem formulation
  #[error("Invalid problem: {message}")]
  InvalidProblem { message: String },

  /// Matrix dimension mismatch
  #[error("Dimension mismatch: expected {expected}, got {actual}")]
  DimensionMismatch { expected: String, actual: String },
}
