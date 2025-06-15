//! Common traits and types for optimization solvers using argmin framework

use cova_algebra::tensors::{DMatrix, DVector};

use crate::{SolverError, SolverResult};

/// Solution information returned by solvers
#[derive(Debug, Clone)]
pub struct Solution {
  /// The optimal solution vector
  pub x:               DVector<f64>,
  /// The optimal objective value
  pub objective_value: f64,
  /// Number of iterations taken
  pub iterations:      u64,
  /// Whether the solver converged
  pub converged:       bool,
  /// Termination reason
  pub termination:     String,
}

/// Common interface for optimization problems that can be solved with argmin
pub trait OptimizationProblem {
  /// Get the problem dimension
  fn dimension(&self) -> usize;

  /// Solve the optimization problem using argmin
  fn solve(&self) -> SolverResult<Solution>;
}
