# Cova Solvers

A Rust library for convex optimization solvers, including linear programming and ADMM (Alternating Direction Method of Multipliers).

## Features

- **Linear Programming**: Simplex method for solving linear programming problems
- **ADMM**: Alternating Direction Method of Multipliers for convex optimization
- **Quadratic Programming**: Support for quadratic programming problems via ADMM
- **Basis Pursuit**: L1-minimization for sparse solutions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
cova-solvers = "0.1.0"
```

## Quick Start

### Linear Programming with Simplex Method

```rust
use cova_solvers::linear_programming::SimplexSolver;
use cova_solvers::Solver;
use nalgebra::{DMatrix, DVector};

// Minimize c^T x subject to Ax <= b, x >= 0
let c = DVector::from_vec(vec![-1.0, -2.0]); // coefficients to minimize
let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 1.0]); // constraint matrix
let b = DVector::from_vec(vec![3.0, 4.0]); // constraint bounds

let mut solver = SimplexSolver::new();
match solver.solve(&c, &a, &b) {
    Ok(solution) => {
        println!("Optimal solution: {:?}", solution.x);
        println!("Optimal value: {}", solution.objective_value);
        println!("Converged: {}", solution.converged);
        println!("Iterations: {}", solution.iterations);
    }
    Err(e) => println!("Error: {}", e),
}
```

### Quadratic Programming with ADMM

```rust
use cova_solvers::admm::{AdmmSolver, AdmmParams};
use nalgebra::{DMatrix, DVector};

// Minimize (1/2) x^T P x + q^T x subject to x >= 0
let p = DMatrix::identity(2, 2); // quadratic term
let q = DVector::from_vec(vec![-1.0, -2.0]); // linear term
let a = DMatrix::zeros(0, 2); // no equality constraints
let b = DVector::zeros(0);

let mut solver = AdmmSolver::new();
match solver.solve_qp(&p, &q, &a, &b) {
    Ok(solution) => {
        println!("Optimal solution: {:?}", solution.x);
        println!("Optimal value: {}", solution.objective_value);
    }
    Err(e) => println!("Error: {}", e),
}
```

### Basis Pursuit (L1-minimization)

```rust
use cova_solvers::admm::AdmmSolver;
use nalgebra::{DMatrix, DVector};

// Minimize ||x||_1 subject to Ax = b
let a = DMatrix::from_row_slice(2, 4, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
let b = DVector::from_vec(vec![1.0, 1.0]);

let mut solver = AdmmSolver::new();
match solver.solve_basis_pursuit(&a, &b) {
    Ok(solution) => {
        println!("Sparse solution: {:?}", solution.x);
        println!("L1 norm: {}", solution.objective_value);
    }
    Err(e) => println!("Error: {}", e),
}
```

## Algorithms

### Simplex Method

The simplex method is implemented for solving linear programming problems of the form:

```
minimize    c^T x
subject to  Ax <= b
            x >= 0
```

Features:
- Standard two-phase simplex algorithm
- Handles infeasible and unbounded problems
- Configurable tolerance and maximum iterations

### ADMM (Alternating Direction Method of Multipliers)

ADMM is implemented for solving convex optimization problems that can be decomposed into:

```
minimize    f(x) + g(z)
subject to  Ax + Bz = c
```

Specific implementations include:
- **Quadratic Programming**: For problems with quadratic objectives
- **Basis Pursuit**: For L1-regularized problems (sparse solutions)
- **Adaptive penalty parameter**: Automatic adjustment of the penalty parameter ρ

## Configuration

### Simplex Solver Parameters

```rust
let mut solver = SimplexSolver::with_params(1e-8, 2000); // tolerance, max_iterations
solver.set_tolerance(1e-6);
solver.set_max_iterations(1000);
```

### ADMM Parameters

```rust
use cova_solvers::admm::AdmmParams;

let params = AdmmParams {
    rho: 1.0,                    // penalty parameter
    primal_tolerance: 1e-6,      // primal residual tolerance
    dual_tolerance: 1e-6,        // dual residual tolerance
    max_iterations: 1000,        // maximum iterations
    alpha: 1.0,                  // over-relaxation parameter
};

let mut solver = AdmmSolver::with_params(params);
```

## Error Handling

The library provides comprehensive error handling through the `SolverError` enum:

- `Infeasible`: Problem has no solution
- `Unbounded`: Objective can be made arbitrarily good
- `ConvergenceFailure`: Solver didn't converge within max iterations
- `NumericalError`: Numerical issues during computation
- `InvalidProblem`: Invalid problem formulation
- `DimensionMismatch`: Matrix/vector dimension mismatches

## Performance Notes

- Both solvers use efficient linear algebra operations via `nalgebra`
- ADMM benefits from matrix factorization caching for repeated solves
- For large problems, consider adjusting tolerances and iteration limits
- The simplex method is exact but can be slower for very large problems
- ADMM is iterative and scales well but provides approximate solutions

## Contributing

Contributions are welcome! Please see our contributing guidelines for more information.

## License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.
