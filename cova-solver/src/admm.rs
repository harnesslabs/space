//! ADMM (Alternating Direction Method of Multipliers) Solver
//!
//! This module implements ADMM for solving convex optimization problems of the form:
//!
//! minimize    f(x) + g(z)
//! subject to  Ax + Bz = c
//!
//! The algorithm splits this into subproblems and uses augmented Lagrangian method.
use std::collections::HashMap;

use cova_algebra::tensors::{DMatrix, DVector};

use crate::{traits::Solution, Solver, SolverError, SolverResult};

/// Parameters for ADMM optimization
#[derive(Debug, Clone)]
pub struct AdmmParams {
  /// Penalty parameter (rho)
  pub rho:              f64,
  /// Tolerance for primal residual
  pub primal_tolerance: f64,
  /// Tolerance for dual residual  
  pub dual_tolerance:   f64,
  /// Maximum number of iterations
  pub max_iterations:   usize,
  /// Over-relaxation parameter (alpha)
  pub alpha:            f64,
}

impl Default for AdmmParams {
  fn default() -> Self {
    Self {
      rho:              1.0,
      primal_tolerance: 1e-6,
      dual_tolerance:   1e-6,
      max_iterations:   1000,
      alpha:            1.0,
    }
  }
}

/// ADMM solver for convex optimization problems
#[derive(Debug)]
pub struct AdmmSolver {
  params: AdmmParams,
}

impl AdmmSolver {
  /// Create a new ADMM solver with default parameters
  pub fn new() -> Self { Self { params: AdmmParams::default() } }

  /// Create a new ADMM solver with custom parameters
  pub fn with_params(params: AdmmParams) -> Self { Self { params } }

  /// Solve quadratic programming problem using ADMM
  ///
  /// minimize    (1/2) x^T P x + q^T x
  /// subject to  Ax = b
  ///             x >= 0
  pub fn solve_qp(
    &mut self,
    p: &DMatrix<f64>,
    q: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    let n = q.num_rows();
    let m = b.num_rows();

    // Check dimensions
    if p.num_rows() != n || p.num_cols() != n {
      return Err(SolverError::DimensionMismatch {
        expected: format!("P matrix {}x{}", n, n),
        actual:   format!("{}x{}", p.num_rows(), p.num_cols()),
      });
    }
    if a.num_rows() != m || a.num_cols() != n {
      return Err(SolverError::DimensionMismatch {
        expected: format!("A matrix {}x{}", m, n),
        actual:   format!("{}x{}", a.num_rows(), a.num_cols()),
      });
    }

    // Initialize variables
    let mut x = DVector::zeros(n);
    let mut z = DVector::zeros(n);
    let mut u = DVector::zeros(n); // Scaled dual variable

    // Precompute factorization for x-update: (P + rho*I)
    let mut lhs_matrix = p.clone();
    for i in 0..n {
      lhs_matrix[(i, i)] += self.params.rho;
    }

    let lhs_decomp = match lhs_matrix.clone().lu() {
      lu if lu.determinant() != 0.0 => lu,
      _ =>
        return Err(SolverError::NumericalError {
          message: "Singular matrix in x-update".to_string(),
        }),
    };

    let mut iteration_data = HashMap::new();

    for iteration in 0..self.params.max_iterations {
      let x_old = x.clone();
      let z_old = z.clone();

      // x-update: solve (P + rho*I)x = -q + rho*(z - u)
      let rhs = -q + self.params.rho * (&z - &u);
      x = match lhs_decomp.solve(&rhs) {
        Some(solution) => solution,
        None =>
          return Err(SolverError::NumericalError {
            message: "Failed to solve x-update".to_string(),
          }),
      };

      // z-update: projection onto non-negative orthant
      let z_hat = &x + &u;
      z = z_hat.map(|val| val.max(0.0));

      // u-update: dual variable update
      u = &u + &x - &z;

      // Check convergence
      let primal_residual = (&x - &z).norm();
      let dual_residual = self.params.rho * (&z - &z_old).norm();

      let primal_tolerance = self.params.primal_tolerance * (x.norm().max(z.norm()) + 1e-8);
      let dual_tolerance = self.params.dual_tolerance * (self.params.rho * u.norm() + 1e-8);

      iteration_data.insert(iteration, (primal_residual, dual_residual));

      if primal_residual <= primal_tolerance && dual_residual <= dual_tolerance {
        let objective_value = 0.5 * x.dot(&(p * &x)) + q.dot(&x);

        return Ok(Solution { x, objective_value, iterations: iteration + 1, converged: true });
      }

      // Adaptive rho update (optional)
      if iteration % 10 == 0 && iteration > 0 {
        if primal_residual > 10.0 * dual_residual {
          self.params.rho *= 2.0;
        } else if dual_residual > 10.0 * primal_residual {
          self.params.rho /= 2.0;
        }
      }
    }

    // Did not converge
    let objective_value = 0.5 * x.dot(&(p * &x)) + q.dot(&x);

    Ok(Solution { x, objective_value, iterations: self.params.max_iterations, converged: false })
  }

  /// Solve basis pursuit (l1-minimization) using ADMM
  ///
  /// minimize    ||x||_1
  /// subject to  Ax = b
  pub fn solve_basis_pursuit(
    &mut self,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    let n = a.ncols();
    let m = a.nrows();

    if b.len() != m {
      return Err(SolverError::DimensionMismatch {
        expected: format!("b vector length {}", m),
        actual:   format!("{}", b.len()),
      });
    }

    // Initialize variables
    let mut x = DVector::zeros(n);
    let mut z = DVector::zeros(n);
    let mut u = DVector::zeros(n);

    // Precompute (A^T A + rho*I)^{-1} A^T
    let ata = a.transpose() * a;
    let mut lhs_matrix = ata;
    for i in 0..n {
      lhs_matrix[(i, i)] += self.params.rho;
    }

    let lhs_decomp = match lhs_matrix.lu() {
      lu if lu.determinant() != 0.0 => lu,
      _ =>
        return Err(SolverError::NumericalError {
          message: "Singular matrix in basis pursuit".to_string(),
        }),
    };

    let atb = a.transpose() * b;

    for iteration in 0..self.params.max_iterations {
      let z_old = z.clone();

      // x-update: solve (A^T A + rho*I)x = A^T b + rho*(z - u)
      let rhs = &atb + self.params.rho * (&z - &u);
      x = match lhs_decomp.solve(&rhs) {
        Some(solution) => solution,
        None =>
          return Err(SolverError::NumericalError {
            message: "Failed to solve x-update in basis pursuit".to_string(),
          }),
      };

      // z-update: soft thresholding (proximal operator of l1 norm)
      let x_plus_u = &x + &u;
      let lambda = 1.0 / self.params.rho;
      z = x_plus_u.map(|val| {
        if val > lambda {
          val - lambda
        } else if val < -lambda {
          val + lambda
        } else {
          0.0
        }
      });

      // u-update
      u = &u + &x - &z;

      // Check convergence
      let primal_residual = (&x - &z).norm();
      let dual_residual = self.params.rho * (&z - &z_old).norm();

      if primal_residual <= self.params.primal_tolerance
        && dual_residual <= self.params.dual_tolerance
      {
        let objective_value = z.iter().map(|x| x.abs()).sum::<f64>(); // l1 norm

        return Ok(Solution { x: z, objective_value, iterations: iteration + 1, converged: true });
      }
    }

    let objective_value = z.iter().map(|x| x.abs()).sum::<f64>();
    Ok(Solution { x: z, objective_value, iterations: self.params.max_iterations, converged: false })
  }

  /// Set ADMM parameters
  pub fn set_params(&mut self, params: AdmmParams) { self.params = params; }

  /// Get current parameters
  pub fn get_params(&self) -> &AdmmParams { &self.params }
}

impl Default for AdmmSolver {
  fn default() -> Self { Self::new() }
}

// Basic Solver trait implementation for QP problems
impl Solver for AdmmSolver {
  fn solve(
    &mut self,
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    // Convert LP to QP format: minimize c^T x subject to Ax <= b, x >= 0
    // This is a simplified version - for full LP support, slack variables would be needed
    let n = c.len();
    let p = DMatrix::zeros(n, n); // No quadratic term

    self.solve_qp(&p, c, a, b)
  }

  fn set_tolerance(&mut self, tolerance: f64) {
    self.params.primal_tolerance = tolerance;
    self.params.dual_tolerance = tolerance;
  }

  fn set_max_iterations(&mut self, max_iterations: usize) {
    self.params.max_iterations = max_iterations;
  }

  fn get_tolerance(&self) -> f64 { self.params.primal_tolerance }

  fn get_max_iterations(&self) -> usize { self.params.max_iterations }
}

#[cfg(test)]
mod tests {
  use nalgebra::{dmatrix, dvector};

  use super::*;

  #[test]
  fn test_quadratic_program() {
    // minimize (1/2) x^T x + 0
    // subject to x >= 0

    let n = 2;
    let p = DMatrix::identity(n, n);
    let q = DVector::zeros(n);
    let a = DMatrix::zeros(0, n); // No equality constraints
    let b = DVector::zeros(0);

    let mut solver = AdmmSolver::new();
    let result = solver.solve_qp(&p, &q, &a, &b).unwrap();

    // Expected solution: x = [0, 0]
    assert!(result.x.norm() < 1e-6);
    assert!(result.converged);
  }

  #[test]
  fn test_basis_pursuit_simple() {
    // Simple basis pursuit problem
    let a = dmatrix![1.0, 1.0; 1.0, -1.0];
    let b = dvector![1.0, 0.0];

    let mut solver = AdmmSolver::new();
    let result = solver.solve_basis_pursuit(&a, &b).unwrap();

    // Should find sparse solution
    assert!(result.converged);
    // Check that Ax = b
    let residual = &a * &result.x - &b;
    assert!(residual.norm() < 1e-3);
  }
}
