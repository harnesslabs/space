//! Linear Programming Solvers
//!
//! This module implements the Simplex method for solving linear programming problems
//! of the form:
//!
//! minimize    c^T x
//! subject to  Ax <= b
//!             x >= 0

use cova_algebra::tensors::{DMatrix, DVector};

use crate::{traits::Solution, Solver, SolverError, SolverResult};

/// Simplex method solver for linear programming
#[derive(Debug)]
pub struct SimplexSolver {
  tolerance:      f64,
  max_iterations: usize,
}

impl SimplexSolver {
  /// Create a new SimplexSolver with default parameters
  pub fn new() -> Self { Self { tolerance: 1e-6, max_iterations: 1000 } }

  /// Create a new SimplexSolver with custom parameters
  pub fn with_params(tolerance: f64, max_iterations: usize) -> Self {
    Self { tolerance, max_iterations }
  }

  /// Convert the problem to standard form and solve using simplex tableau
  fn solve_standard_form(
    &self,
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    let m = a.nrows();
    let n = a.ncols();

    // Check dimensions
    if b.len() != m {
      return Err(SolverError::DimensionMismatch {
        expected: format!("b vector length {}", m),
        actual:   format!("{}", b.len()),
      });
    }
    if c.len() != n {
      return Err(SolverError::DimensionMismatch {
        expected: format!("c vector length {}", n),
        actual:   format!("{}", c.len()),
      });
    }

    // Check for negative b values (infeasible basic solution)
    if b.iter().any(|&x| x < 0.0) {
      return Err(SolverError::Infeasible);
    }

    // Create tableau: [A I b; c^T 0 0]
    let mut tableau = DMatrix::zeros(m + 1, n + m + 1);

    // Fill constraint matrix A
    tableau.view_mut((0, 0), (m, n)).copy_from(a);

    // Add identity matrix for slack variables
    for i in 0..m {
      tableau[(i, n + i)] = 1.0;
    }

    // Add RHS vector b
    for i in 0..m {
      tableau[(i, n + m)] = b[i];
    }

    // Add objective function coefficients (negated for maximization)
    for j in 0..n {
      tableau[(m, j)] = c[j];
    }

    // Basic variables start as slack variables
    let mut basic_vars: Vec<usize> = (n..n + m).collect();

    // Simplex iterations
    for iteration in 0..self.max_iterations {
      // Find entering variable (most negative in objective row)
      let mut entering_col = None;
      let mut min_ratio = 0.0;

      for j in 0..n + m {
        let obj_coeff = tableau[(m, j)];
        if obj_coeff < -self.tolerance {
          if entering_col.is_none() || obj_coeff < min_ratio {
            entering_col = Some(j);
            min_ratio = obj_coeff;
          }
        }
      }

      // If no entering variable, we're optimal
      let entering_col = match entering_col {
        Some(col) => col,
        None => {
          // Extract solution
          let mut x = DVector::zeros(n);
          for (i, &basic_var) in basic_vars.iter().enumerate() {
            if basic_var < n {
              x[basic_var] = tableau[(i, n + m)];
            }
          }
          let objective_value = -tableau[(m, n + m)];

          return Ok(Solution { x, objective_value, iterations: iteration, converged: true });
        },
      };

      // Find leaving variable (minimum ratio test)
      let mut leaving_row = None;
      let mut min_ratio = f64::INFINITY;

      for i in 0..m {
        let pivot_elem = tableau[(i, entering_col)];
        if pivot_elem > self.tolerance {
          let ratio = tableau[(i, n + m)] / pivot_elem;
          if ratio >= 0.0 && ratio < min_ratio {
            min_ratio = ratio;
            leaving_row = Some(i);
          }
        }
      }

      // If no leaving variable, problem is unbounded
      let leaving_row = match leaving_row {
        Some(row) => row,
        None => return Err(SolverError::Unbounded),
      };

      // Perform pivot operation
      let pivot = tableau[(leaving_row, entering_col)];
      if pivot.abs() < self.tolerance {
        return Err(SolverError::NumericalError { message: "Pivot element too small".to_string() });
      }

      // Scale pivot row
      for j in 0..=n + m {
        tableau[(leaving_row, j)] /= pivot;
      }

      // Eliminate column
      for i in 0..=m {
        if i != leaving_row {
          let factor = tableau[(i, entering_col)];
          for j in 0..=n + m {
            tableau[(i, j)] -= factor * tableau[(leaving_row, j)];
          }
        }
      }

      // Update basic variables
      basic_vars[leaving_row] = entering_col;
    }

    Err(SolverError::ConvergenceFailure { max_iterations: self.max_iterations })
  }
}

impl Default for SimplexSolver {
  fn default() -> Self { Self::new() }
}

impl Solver for SimplexSolver {
  fn solve(
    &mut self,
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    self.solve_standard_form(c, a, b)
  }

  fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }

  fn set_max_iterations(&mut self, max_iterations: usize) { self.max_iterations = max_iterations; }

  fn get_tolerance(&self) -> f64 { self.tolerance }

  fn get_max_iterations(&self) -> usize { self.max_iterations }
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

    let mut solver = SimplexSolver::new();
    let result = solver.solve(&c, &a, &b).unwrap();

    // Expected solution: x1 = 1, x2 = 2, objective = -5
    assert!((result.x[0] - 1.0).abs() < 1e-6);
    assert!((result.x[1] - 2.0).abs() < 1e-6);
    assert!((result.objective_value - (-5.0)).abs() < 1e-6);
    assert!(result.converged);
  }
}
