//! Linear Programming using argmin framework
//!
//! This module implements linear programming solvers using argmin's optimization framework.
//! We transform the linear program into a form suitable for gradient-based optimization.

use argmin::{
  core::{CostFunction, Executor, Gradient},
  solver::{gradientdescent::SteepestDescent, linesearch::BacktrackingLineSearch},
};
use argmin_math::ArgminDot;
use cova_algebra::tensors::{DMatrix, DVector};

use crate::{
  traits::{OptimizationProblem, Solution},
  SolverError, SolverResult,
};

/// Linear programming problem: minimize c^T x subject to Ax <= b, x >= 0
#[derive(Debug, Clone)]
pub struct LinearProgram {
  /// Objective function coefficients
  pub c:              DVector<f64>,
  /// Constraint matrix A
  pub a:              DMatrix<f64>,
  /// Constraint bounds b
  pub b:              DVector<f64>,
  /// Solver tolerance
  pub tolerance:      f64,
  /// Maximum iterations
  pub max_iterations: u64,
}

impl LinearProgram {
  /// Create a new linear programming problem
  pub fn new(c: DVector<f64>, a: DMatrix<f64>, b: DVector<f64>) -> SolverResult<Self> {
    // Validate dimensions
    let n = c.len();
    let m = b.len();

    if a.nrows() != m {
      return Err(SolverError::DimensionMismatch {
        expected: format!("A matrix rows {}", m),
        actual:   format!("{}", a.nrows()),
      });
    }

    if a.ncols() != n {
      return Err(SolverError::DimensionMismatch {
        expected: format!("A matrix cols {}", n),
        actual:   format!("{}", a.ncols()),
      });
    }

    Ok(Self { c, a, b, tolerance: 1e-6, max_iterations: 1000 })
  }

  /// Set solver tolerance
  pub fn with_tolerance(mut self, tolerance: f64) -> Self {
    self.tolerance = tolerance;
    self
  }

  /// Set maximum iterations
  pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
    self.max_iterations = max_iterations;
    self
  }
}

/// Cost function for linear programming with penalty method
/// We minimize: c^T x + penalty * sum(max(0, Ax - b)) + penalty * sum(max(0, -x))
impl CostFunction for LinearProgram {
  type Output = f64;
  type Param = DVector<f64>;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    let penalty = 1000.0;

    // Original objective
    let objective = self.c.dot(x);

    // Constraint violations: Ax <= b
    let constraint_violations = &self.a * x - &self.b;
    let constraint_penalty: f64 = constraint_violations.iter().map(|&v| v.max(0.0).powi(2)).sum();

    // Non-negativity violations: x >= 0
    let nonnegativity_penalty: f64 = x.iter().map(|&v| (-v).max(0.0).powi(2)).sum();

    Ok(objective + penalty * (constraint_penalty + nonnegativity_penalty))
  }
}

/// Gradient for the linear programming cost function
impl Gradient for LinearProgram {
  type Gradient = DVector<f64>;
  type Param = DVector<f64>;

  fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let penalty = 1000.0;
    let mut grad = self.c.clone();

    // Gradient from constraint violations: Ax <= b
    let constraint_violations = &self.a * x - &self.b;
    for (i, &violation) in constraint_violations.iter().enumerate() {
      if violation > 0.0 {
        for j in 0..x.len() {
          grad[j] += 2.0 * penalty * violation * self.a[(i, j)];
        }
      }
    }

    // Gradient from non-negativity violations: x >= 0
    for (i, &xi) in x.iter().enumerate() {
      if xi < 0.0 {
        grad[i] += -2.0 * penalty * xi;
      }
    }

    Ok(grad)
  }
}

impl OptimizationProblem for LinearProgram {
  fn dimension(&self) -> usize { self.c.len() }

  fn solve(&self) -> SolverResult<Solution> {
    // Initial guess: zeros
    let init_param = DVector::zeros(self.dimension());

    // Set up line search
    let linesearch = BacktrackingLineSearch::new();

    // Set up solver
    let solver = SteepestDescent::new(linesearch);

    // Set up executor
    let res = Executor::new(self.clone(), solver)
      .configure(|state| state.param(init_param).max_iters(self.max_iterations).target_cost(0.0))
      .run();

    match res {
      Ok(result) => {
        let state = result.state();
        let x = state.best_param.clone().unwrap();
        let objective_value = self.c.dot(&x);
        let iterations = state.iter;
        let converged = state.cost < self.tolerance;
        let termination = format!("{:?}", state.termination_reason);

        Ok(Solution { x, objective_value, iterations, converged, termination })
      },
      Err(e) =>
        Err(SolverError::NumericalError { message: format!("Argmin solver failed: {}", e) }),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_simple_lp() {
    // minimize -x1 - 2*x2
    // subject to x1 + x2 <= 3
    //           2*x1 + x2 <= 4
    //           x1, x2 >= 0

    let c = DVector::from_vec(vec![-1.0, -2.0]);
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 1.0]);
    let b = DVector::from_vec(vec![3.0, 4.0]);

    let lp = LinearProgram::new(c, a, b).unwrap();
    let result = lp.solve().unwrap();

    // Check that solution is reasonable (may not be exact due to penalty method)
    assert!(result.x[0] >= -0.1); // x1 >= 0 (with tolerance)
    assert!(result.x[1] >= -0.1); // x2 >= 0 (with tolerance)

    // Check constraints are approximately satisfied
    let constraint1 = result.x[0] + result.x[1];
    let constraint2 = 2.0 * result.x[0] + result.x[1];
    assert!(constraint1 <= 3.1); // Allow some tolerance
    assert!(constraint2 <= 4.1);
  }
}
