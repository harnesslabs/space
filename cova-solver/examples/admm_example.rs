//! ADMM Examples using Clarabel
//!
//! This example demonstrates how to use the ADMM solver with Clarabel
//! for solving various convex optimization problems.

use cova_algebra::tensors::{DMatrix, DVector};
use cova_solver::{
  admm::{AdmmParams, AdmmSolver},
  SolverResult,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("ADMM Examples using Clarabel\n");

  // Example 1: LASSO regression
  println!("=== Example 1: LASSO Regression ===");
  lasso_example()?;

  // Example 2: Basis pursuit
  println!("\n=== Example 2: Basis Pursuit ===");
  basis_pursuit_example()?;

  // Example 3: Custom ADMM problem
  println!("\n=== Example 3: Custom ADMM Problem ===");
  custom_admm_example()?;

  Ok(())
}

/// Example 1: LASSO regression
/// minimize ||Ax - b||² + λ||x||₁
fn lasso_example() -> SolverResult<()> {
  // Create a simple regression problem
  let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 1.0, 1.0, 1.0]);
  let b = DVector::from_vec(vec![3.0, 4.0, 2.0]);
  let lambda = 0.1; // Regularization parameter

  println!("Problem: minimize ||Ax - b||² + λ||x||₁");
  println!("A = \n{}", a);
  println!("b = {:?}", b.as_slice());
  println!("λ = {}", lambda);

  let mut solver = AdmmSolver::new();
  let solution = solver.solve_lasso(&a, &b, lambda)?;

  println!("Solution:");
  println!("  x = {:?}", solution.x.as_slice());
  println!("  Objective value = {:.6}", solution.objective_value);
  println!("  Iterations = {}", solution.iterations);
  println!("  Converged = {}", solution.converged);

  // Verify the solution
  let residual = &a * &solution.x - &b;
  let residual_norm = residual.norm();
  let l1_norm: f64 = solution.x.iter().map(|x| x.abs()).sum();

  println!("  Residual norm = {:.6}", residual_norm);
  println!("  L1 norm = {:.6}", l1_norm);

  Ok(())
}

/// Example 2: Basis pursuit (sparse recovery)
/// minimize ||x||₁ subject to Ax = b
fn basis_pursuit_example() -> SolverResult<()> {
  // Create an underdetermined system (more variables than equations)
  let a = DMatrix::from_row_slice(2, 4, &[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
  let b = DVector::from_vec(vec![1.0, 2.0]);

  println!("Problem: minimize ||x||₁ subject to Ax = b");
  println!("A = \n{}", a);
  println!("b = {:?}", b.as_slice());

  let mut solver = AdmmSolver::new();
  let solution = solver.solve_basis_pursuit(&a, &b)?;

  println!("Solution:");
  println!("  x = {:?}", solution.x.as_slice());
  println!("  Objective value (L1 norm) = {:.6}", solution.objective_value);
  println!("  Iterations = {}", solution.iterations);
  println!("  Converged = {}", solution.converged);

  // Verify constraint satisfaction
  let constraint_residual = &a * &solution.x - &b;
  println!("  Constraint residual = {:.6}", constraint_residual.norm());

  Ok(())
}

/// Example 3: Custom ADMM problem with box constraints
/// minimize (1/2)||x - x₀||² subject to x ∈ [0, 1]ⁿ
fn custom_admm_example() -> SolverResult<()> {
  let n = 3;
  let x0 = DVector::from_vec(vec![1.5, -0.5, 2.0]); // Target point outside [0,1]³

  println!("Problem: minimize (1/2)||x - x₀||² subject to x ∈ [0,1]ⁿ");
  println!("x₀ = {:?}", x0.as_slice());

  // Set up ADMM problem:
  // minimize (1/2)||x - x₀||² + indicator_[0,1](z)
  // subject to x = z

  let p = DMatrix::identity(n, n);
  let q = -&x0;
  let a_constraint = DMatrix::identity(n, n);
  let b_constraint = -DMatrix::identity(n, n);
  let c = DVector::zeros(n);

  // z-update: projection onto [0,1]ⁿ
  let z_update = |target: &DVector<f64>, _z_old: &DVector<f64>, _rho: f64| {
    target.map(|val| val.max(0.0).min(1.0))
  };

  let mut solver = AdmmSolver::new();
  let solution = solver.solve_qp_admm(&p, &q, &a_constraint, &b_constraint, &c, z_update)?;

  println!("Solution:");
  println!("  x = {:?}", solution.x.as_slice());
  println!("  Objective value = {:.6}", solution.objective_value);
  println!("  Iterations = {}", solution.iterations);
  println!("  Converged = {}", solution.converged);

  // Verify box constraints
  let in_bounds = solution.x.iter().all(|&x| x >= -1e-6 && x <= 1.0 + 1e-6);
  println!("  Satisfies box constraints = {}", in_bounds);

  Ok(())
}
