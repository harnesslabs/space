# Algebra Crate

A Rust library providing algebraic structures and operations, with a focus on modular arithmetic and abstract algebra concepts.

[![Crates.io - harness-algebra](https://img.shields.io/crates/v/harness-algebra?label=harness-algebra)](https://crates.io/crates/harness-algebra)
[![docs.rs - harness-algebra](https://img.shields.io/docsrs/harness-algebra?label=docs.rs%20harness-algebra)](https://docs.rs/harness-algebra)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Features

- **Modular Arithmetic**: Create custom modular number types with the `modular!` macro
- **Abstract Algebra**: Implementations of fundamental algebraic structures:
  - Groups (both Abelian and Non-Abelian)
  - Rings
  - Fields
  - Modules
  - Vector Spaces

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
harness-algebra = "*"
```

### Modular Arithmetic

Create a new modular number type using the `modular!` macro:

```rust
use harness_algebra::{group::Group, modular, ring::Ring};

// Create a type for numbers modulo 7
modular!(Mod7, u32, 7);

let a = Mod7::new(3);
let b = Mod7::new(5);

// Addition: 3 + 5 = 8 ≡ 1 (mod 7)
let sum = a + b;
assert_eq!(sum.value(), 1);

// Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
let product = a * b;
assert_eq!(product.value(), 1);
```

### Vector Spaces

Create and manipulate vectors over any field, such as the finite field of integers modulo 7:

```rust
use harness_algebra::{vector::{Vector, VectorSpace}, ring::Field, modular};

modular!(Mod7, u32, 7);
prime_field!(Mod7);
impl Field for Mod7 {
    fn multiplicative_inverse(&self) -> Self {
        todo!("Implement multiplicative inverse for Mod7")
    }
}

let v1 = Vector::<3, Mod7>([Mod7::new(1), Mod7::new(2), Mod7::new(3)]);
let v2 = Vector::<3, Mod7>([Mod7::new(4), Mod7::new(5), Mod7::new(6)]);
let sum = v1 + v2;
```

## Documentation

The complete API documentation is available on [docs.rs](https://docs.rs/algebra).

### Modules

- [`arithmetic`](https://docs.rs/harness-algebra/latest/harness_algebra/arithmetic/index.html): Basic arithmetic traits and operations
- [`group`](https://docs.rs/harness-algebra/latest/harness_algebra/group/index.html): Group theory abstractions and implementations
- [`ring`](https://docs.rs/harness-algebra/latest/harness_algebra/ring/index.html): Ring theory abstractions and implementations
- [`module`](https://docs.rs/harness-algebra/latest/harness_algebra/module/index.html): Module theory abstractions and implementations
- [`vector`](https://docs.rs/harness-algebra/latest/harness_algebra/vector/index.html): Vector space abstractions and implementations
- [`modular`](https://docs.rs/harness-algebra/latest/harness_algebra/modular/index.html): Modular arithmetic abstractions and implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](../LICENSE) file for details.

## Examples

```rust
use algebra::{Group, Ring};

modular!(Mod7, u32, 7);

// Group operations
let a = Mod7::new(3);
let inverse = a.inverse();  // 4 (mod 7)
let identity = Mod7::identity();  // 0 (mod 7)

// Ring operations
let one = Mod7::one();  // 1 (mod 7)
let zero = Mod7::zero();  // 0 (mod 7)
```


