# Algebra Crate

A Rust library providing algebraic structures and operations, with a focus on modular arithmetic and abstract algebra concepts.

## Features

- **Modular Arithmetic**: Create custom modular number types with the `modular!` macro
- **Abstract Algebra**: Implementations of fundamental algebraic structures:
  - Groups (both Abelian and Non-Abelian)
  - Rings
  - Fields

## Usage

### Modular Arithmetic

Create a new modular number type using the `modular!` macro:

```rust
modular!(Mod7, u32, 7);  // Creates a type for numbers modulo 7

let a = Mod7::new(3);
let b = Mod7::new(5);
let sum = a + b;  // 8 ≡ 1 (mod 7)
```

### Algebraic Structures

The crate provides traits for various algebraic structures:

- `Group`: Basic group operations with identity and inverse
- `AbelianGroup`: Commutative groups with addition operations
- `NonAbelianGroup`: Non-commutative groups with multiplication operations
- `Ring`: Structures with both addition and multiplication
- `Field`: Rings with multiplicative inverses

## Dependencies

- `num`: For numeric traits and operations

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


