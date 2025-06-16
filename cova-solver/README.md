# Cova Solver

A Rust library providing optimization solvers for convex problems, built on top of Clarabel.

## Features

- **ADMM (Alternating Direction Method of Multipliers)**: Uses Clarabel to solve convex subproblems
- **LASSO Regression**: Sparse regression with L1 regularization
- **Basis Pursuit**: Sparse recovery for underdetermined systems
- **Custom ADMM Problems**: Flexible framework for defining custom optimization problems

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
cova-solver = { path = "." }
cova-algebra = { path = "../cova-algebra" }
```

## Usage

### LASSO Regression

Solve sparse regression problems with L1 regularization:

```rust
use cova_algebra::tensors::{DMatrix, DVector};
use cova_solver::admm::AdmmSolver;

// Problem: minimize ||Ax - b||² + λ||x||₁
let a = DMatrix::from_row_slice(3, 2, &[
    1.0, 2.0,
    2.0, 1.0,
    1.0, 1.0,
]);
let b = DVector::from_vec(vec![3.0, 4.0, 2.0]);
let lambda = 0.1; // Regularization parameter

let mut solver = AdmmSolver::new();
let solution = solver.solve_lasso(&a, &b, lambda)?;

println!("Solution: {:?}", solution.x.as_slice());
println!("Converged: {}", solution.converged);
```

### Basis Pursuit

Solve sparse recovery problems:

```rust
use cova_algebra::tensors::{DMatrix, DVector};
use cova_solver::admm::AdmmSolver;

// Problem: minimize ||x||₁ subject to Ax = b
let a = DMatrix::from_row_slice(2, 4, &[
    1.0, 1.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0,
]);
let b = DVector::from_vec(vec![1.0, 2.0]);

let mut solver = AdmmSolver::new();
let solution = solver.solve_basis_pursuit(&a, &b)?;

// The solution should be sparse and satisfy Ax = b
```

### Custom ADMM Problems

Define custom optimization problems using the general ADMM framework:

```rust
use cova_algebra::tensors::{DMatrix, DVector};
use cova_solver::admm::{AdmmSolver, AdmmParams};

// Example: Box-constrained quadratic programming
// minimize (1/2)||x - x₀||² subject to x ∈ [0,1]ⁿ

let n = 3;
let x0 = DVector::from_vec(vec![1.5, -0.5, 2.0]); // Target outside [0,1]³

// Set up ADMM formulation
let p = DMatrix::identity(n, n);
let q = -&x0;
let a_constraint = DMatrix::identity(n, n);
let b_constraint = -DMatrix::identity(n, n);
let c = DVector::zeros(n);

// Define the z-update (projection onto [0,1]ⁿ)
let z_update = |target: &DVector<f64>, _z_old: &DVector<f64>, _rho: f64| {
    target.map(|val| val.max(0.0).min(1.0))
};

let mut solver = AdmmSolver::new();
let solution = solver.solve_qp_admm(&p, &q, &a_constraint, &b_constraint, &c, z_update)?;
```

## ADMM Algorithm

The ADMM solver decomposes problems of the form:

```
minimize    f(x) + g(z)
subject to  Ax + Bz = c
```

into three iterative steps:

1. **x-update**: Minimize f(x) + (ρ/2)||Ax + Bz - c + u||² (solved using Clarabel)
2. **z-update**: Apply proximal operator of g(z) (problem-specific, user-defined)
3. **u-update**: Update dual variable u := u + ρ(Ax + Bz - c)

## Configuration

Customize ADMM parameters:

```rust
use cova_solver::admm::{AdmmSolver, AdmmParams};

let params = AdmmParams {
    rho: 1.0,                    // Penalty parameter
    primal_tolerance: 1e-6,      // Primal residual tolerance
    dual_tolerance: 1e-6,        // Dual residual tolerance
    max_iterations: 1000,        // Maximum iterations
    alpha: 1.0,                  // Over-relaxation parameter
};

let solver = AdmmSolver::with_params(params);
```

## Examples

Run the examples to see the solver in action:

```bash
cargo run --example admm_example
```

This will demonstrate:
- LASSO regression with different regularization parameters
- Basis pursuit for sparse recovery
- Custom box-constrained optimization

## How it Works

The library combines two powerful tools:

1. **Clarabel**: A modern conic optimization solver in pure Rust for solving the convex subproblems
2. **ADMM**: A versatile algorithm that can handle a wide variety of convex optimization problems

Key benefits:
- **Pure Rust**: No C/C++/Fortran dependencies
- **Flexible**: Easy to define custom optimization problems
- **Efficient**: Leverages Clarabel's high-performance interior-point methods
- **Robust**: Well-tested algorithm with proven convergence properties

## Limitations

- Currently focused on convex optimization problems
- The x-update subproblems must be efficiently solvable by Clarabel (quadratic programming)
- For very large-scale problems, specialized solvers might be more efficient

## Dependencies

- `clarabel`: Pure Rust conic optimization solver
- `cova-algebra`: Linear algebra operations
- `thiserror`: Error handling

## License

AGPL-3.0
