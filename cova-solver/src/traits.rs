//! Common traits and types for optimization solvers

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
  pub iterations:      usize,
  /// Whether the solver converged
  pub converged:       bool,
}

/// Common interface for optimization solvers
pub trait Solver {
  /// Solve the optimization problem
  fn solve(
    &mut self,
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution>;

  /// Set solver parameters
  fn set_tolerance(&mut self, tolerance: f64);
  fn set_max_iterations(&mut self, max_iter: usize);

  /// Get solver status
  fn get_tolerance(&self) -> f64;
  fn get_max_iterations(&self) -> usize;
}
