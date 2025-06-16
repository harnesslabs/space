//! Error types for optimization solvers

use thiserror::Error;

/// Result type for solver operations
pub type SolverResult<T> = Result<T, SolverError>;

/// Errors that can occur during optimization
#[derive(Error, Debug, Clone)]
pub enum SolverError {
  /// Dimension mismatch between matrices/vectors
  #[error("Dimension mismatch: expected {expected}, got {actual}")]
  DimensionMismatch { expected: String, actual: String },

  /// Invalid problem specification
  #[error("Invalid problem: {message}")]
  InvalidProblem { message: String },

  /// Numerical error during computation
  #[error("Numerical error: {message}")]
  NumericalError { message: String },

  /// Convergence failure
  #[error("Failed to converge: {message}")]
  ConvergenceError { message: String },

  /// Infeasible problem
  #[error("Problem is infeasible: {message}")]
  InfeasibleProblem { message: String },

  /// Unbounded problem
  #[error("Problem is unbounded: {message}")]
  UnboundedProblem { message: String },
}
