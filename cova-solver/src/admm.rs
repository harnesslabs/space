//! ADMM (Alternating Direction Method of Multipliers) using Clarabel
//!
//! This module implements ADMM by using Clarabel to solve the convex subproblems.
//! ADMM solves problems of the form:
//!
//! minimize    f(x) + g(z)
//! subject to  Ax + Bz = c
//!
//! The algorithm alternates between:
//! 1. x-update: minimize f(x) + (ρ/2)||Ax + Bz - c + u||²
//! 2. z-update: minimize g(z) + (ρ/2)||Ax + Bz - c + u||²
//! 3. u-update: u := u + ρ(Ax + Bz - c)

use clarabel::{algebra::*, solver::*};
use cova_algebra::tensors::{DMatrix, DVector};

use crate::{
  traits::{OptimizationProblem, Solution},
  SolverError, SolverResult,
};

/// ADMM parameters
#[derive(Debug, Clone)]
pub struct AdmmParams {
  /// Penalty parameter (ρ)
  pub rho:              f64,
  /// Primal tolerance
  pub primal_tolerance: f64,
  /// Dual tolerance
  pub dual_tolerance:   f64,
  /// Maximum iterations
  pub max_iterations:   usize,
  /// Over-relaxation parameter (α)
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

/// ADMM solver using Clarabel for subproblems
#[derive(Debug)]
pub struct AdmmSolver {
  params: AdmmParams,
}

impl AdmmSolver {
  /// Create a new ADMM solver
  pub fn new() -> Self { Self { params: AdmmParams::default() } }

  /// Create ADMM solver with custom parameters
  pub fn with_params(params: AdmmParams) -> Self { Self { params } }

  /// Solve quadratic programming with equality constraints using ADMM
  ///
  /// minimize    (1/2) x^T P x + q^T x + g(z)
  /// subject to  Ax + Bz = c
  ///
  /// where g(z) is handled by the z-update (e.g., indicator functions, norms)
  pub fn solve_qp_admm<F>(
    &mut self,
    p: &DMatrix<f64>,
    q: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    c: &DVector<f64>,
    z_update: F,
  ) -> SolverResult<Solution>
  where
    F: Fn(&DVector<f64>, &DVector<f64>, f64) -> DVector<f64>,
  {
    let n = q.len();
    let m = c.len();
    let z_dim = b.ncols();

    // Initialize variables
    let mut x = DVector::zeros(n);
    let mut z = DVector::zeros(z_dim);
    let mut u = DVector::zeros(m); // Dual variable

    // Set up Clarabel solver for x-update
    // The x-update subproblem is:
    // minimize (1/2) x^T P x + q^T x + (ρ/2)||Ax + Bz - c + u||²
    // This becomes: minimize (1/2) x^T (P + ρA^TA) x + (q + ρA^T(Bz - c + u))^T x

    let ata = a.transpose() * a;
    let p_aug = p + self.params.rho * &ata;

    for iteration in 0..self.params.max_iterations {
      let z_old = z.clone();

      // x-update: solve QP using Clarabel
      let rhs = b * &z - c + &u;
      let q_aug = q + self.params.rho * (a.transpose() * &rhs);

      x = self.solve_x_update(&p_aug, &q_aug)?;

      // z-update: apply proximal operator (problem-specific)
      let ax_plus_u = a * &x + &u;
      let z_target = &ax_plus_u + c;
      z = z_update(&z_target, &z_old, self.params.rho);

      // u-update: dual variable update
      let residual = a * &x + b * &z - c;
      u = &u + self.params.rho * &residual;

      // Check convergence
      let primal_residual = residual.norm();
      let dual_residual = self.params.rho * (a.transpose() * (&z - &z_old)).norm();

      if primal_residual <= self.params.primal_tolerance
        && dual_residual <= self.params.dual_tolerance
      {
        let objective_value = 0.5f64.mul_add(x.dot(&(p * &x)), q.dot(&x));
        return Ok(Solution {
          x,
          objective_value,
          iterations: iteration as u64 + 1,
          converged: true,
          termination: "Converged".to_string(),
        });
      }
    }

    let objective_value = 0.5f64.mul_add(x.dot(&(p * &x)), q.dot(&x));
    Ok(Solution {
      x,
      objective_value,
      iterations: self.params.max_iterations as u64,
      converged: false,
      termination: "MaxIterations".to_string(),
    })
  }

  /// Solve the x-update subproblem using Clarabel
  fn solve_x_update(&self, p: &DMatrix<f64>, q: &DVector<f64>) -> SolverResult<DVector<f64>> {
    let n = q.len();

    // Convert dense P matrix to sparse CSC format for Clarabel
    // For now, use a simple dense-to-sparse conversion
    let (col_offsets, row_indices, values) = dense_to_csc(p);
    let p_csc = CscMatrix::new(n, n, col_offsets, row_indices, values);

    let q_vec: Vec<f64> = q.iter().cloned().collect();

    // No constraints for the x-update (it's unconstrained QP)
    let a_csc = CscMatrix::new(0, n, vec![0; n + 1], vec![], vec![]);
    let b_vec: Vec<f64> = Vec::new();
    let cones: Vec<SupportedConeT<f64>> = Vec::new();

    // Set up Clarabel settings
    let settings = DefaultSettingsBuilder::default().max_iter(1000).verbose(false).build().unwrap();

    // Create and solve the problem
    let mut solver =
      DefaultSolver::new(&p_csc, &q_vec, &a_csc, &b_vec, &cones, settings).map_err(|e| {
        SolverError::NumericalError { message: format!("Failed to create Clarabel solver: {e:?}") }
      })?;

    solver.solve();

    let result = solver.solution;
    match result.status {
      SolverStatus::Solved => Ok(DVector::from_vec(result.x)),
      _ => Err(SolverError::NumericalError {
        message: format!("Clarabel failed with status: {:?}", result.status),
      }),
    }
  }

  /// Solve LASSO problem: minimize ||Ax - b||² + λ||x||₁
  pub fn solve_lasso(
    &mut self,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lambda: f64,
  ) -> SolverResult<Solution> {
    let n = a.ncols();

    // LASSO as ADMM:
    // minimize ||Ax - b||² + λ||z||₁
    // subject to x = z

    let p = 2.0 * (a.transpose() * a);
    let q = -2.0 * (a.transpose() * b);
    let a_constraint = DMatrix::identity(n, n);
    let b_constraint = -DMatrix::identity(n, n);
    let c = DVector::zeros(n);

    // z-update for LASSO: soft thresholding
    let z_update = move |target: &DVector<f64>, _z_old: &DVector<f64>, rho: f64| {
      let threshold = lambda / rho;
      target.map(|val| {
        if val > threshold {
          val - threshold
        } else if val < -threshold {
          val + threshold
        } else {
          0.0
        }
      })
    };

    self.solve_qp_admm(&p, &q, &a_constraint, &b_constraint, &c, z_update)
  }

  /// Solve basis pursuit: minimize ||x||₁ subject to Ax = b
  pub fn solve_basis_pursuit(
    &mut self,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
  ) -> SolverResult<Solution> {
    // Use LASSO with very small λ to approximate basis pursuit
    self.solve_lasso(a, b, 1e-8)
  }
}

/// Convert dense matrix to CSC format (col_offsets, row_indices, values)
fn dense_to_csc(matrix: &DMatrix<f64>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
  let nrows = matrix.nrows();
  let ncols = matrix.ncols();

  let mut col_offsets = vec![0];
  let mut row_indices = Vec::new();
  let mut values = Vec::new();

  for j in 0..ncols {
    for i in 0..nrows {
      let val = matrix[(i, j)];
      if val.abs() > 1e-15 {
        row_indices.push(i);
        values.push(val);
      }
    }
    col_offsets.push(row_indices.len());
  }

  (col_offsets, row_indices, values)
}

impl Default for AdmmSolver {
  fn default() -> Self { Self::new() }
}

impl OptimizationProblem for AdmmSolver {
  fn dimension(&self) -> usize {
    // This is problem-dependent, so we'll return 0 as placeholder
    0
  }

  fn solve(&self) -> SolverResult<Solution> {
    Err(SolverError::InvalidProblem {
      message: "ADMM solver requires specific problem setup via solve_lasso or solve_basis_pursuit"
        .to_string(),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_lasso_simple() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let lambda = 0.1;

    let mut solver = AdmmSolver::new();
    let result = solver.solve_lasso(&a, &b, lambda).unwrap();

    // Should find sparse solution close to [1.0, 2.0] with some shrinkage
    assert!(result.converged);
    assert!(result.x.len() == 2);
  }

  #[test]
  fn test_basis_pursuit() {
    let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
    let b = DVector::from_vec(vec![1.0]);

    let mut solver = AdmmSolver::new();
    let result = solver.solve_basis_pursuit(&a, &b).unwrap();

    // Should find sparse solution that satisfies Ax = b
    assert!(result.converged);
    let residual = (&a * &result.x - &b).norm();
    assert!(residual < 1e-3);
  }
}
