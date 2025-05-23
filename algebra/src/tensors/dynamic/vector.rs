//! # Dynamic Vector Module
//!
//! This module provides a flexible implementation of vectors with dynamically determined
//! dimensions.
//!
//! ## Mathematical Background
//!
//! A vector space $V$ over a field $F$ is a set equipped with operations of addition and scalar
//! multiplication that satisfy the vector space axioms. This implementation represents elements of
//! $V$ as an ordered collection of components from the field $F$.
//!
//! For any two vectors $\mathbf{u}, \mathbf{v} \in V$ and scalar $\alpha \in F$:
//!
//! - Vector addition: $\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)$
//! - Scalar multiplication: $\alpha\mathbf{v} = (\alpha v_1, \alpha v_2, \ldots, \alpha v_n)$
//! - Additive inverse (negation): $-\mathbf{v} = (-v_1, -v_2, \ldots, -v_n)$
//!
//! ## Zero Vector Handling
//!
//! This implementation represents the zero vector as an empty vector with no components.
//! Any vector with all components equal to zero is also considered a zero vector.
//!
//! ## Features
//!
//! - Generic implementation that works with any field type
//! - Support for vector arithmetic operations (+, -, *, scalar multiplication)
//! - Efficient component access and modification
//! - Implements algebraic traits like `Zero`, `Group`, and `VectorSpace`
//!
//! ## Examples
//!
//! ```
//! use harness_algebra::{prelude::*, tensors::dynamic::vector::DynamicVector};
//!
//! // Create a vector with components [1, 2, 3]
//! let vec1 = DynamicVector::<f64>::from([1.0, 2.0, 3.0]);
//!
//! // Create the zero vector
//! let zero = DynamicVector::<f64>::zero();
//!
//! // Vector addition
//! let vec2 = DynamicVector::<f64>::from([4.0, 5.0, 6.0]);
//! let sum = vec1.clone() + vec2;
//!
//! // Scalar multiplication
//! let scaled = vec1 * 2.0;
//! ```

// TODO (autoparallel): We could use `MaybeUninit` to avoid the `Vec` allocation especially in the
// zero case.

// TODO (autoparallel): We could also have this be generic over an inner vector too and use
// `smallvec` and `tinyvec` if need be.

use std::fmt;

use super::*;
use crate::category::Category;

/// # Dynamic Vector
///
/// A dynamically-sized vector implementation for n-dimensional vector spaces.
///
/// ## Mathematical Background
///
/// A vector space $V$ over a field $F$ is a set equipped with operations of addition and scalar
/// multiplication that satisfy the vector space axioms. This implementation represents elements of
/// $V$ as an ordered collection of components from the field $F$.
///
/// For any two vectors $\mathbf{u}, \mathbf{v} \in V$ and scalar $\alpha \in F$:
///
/// - Vector addition: $\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)$
/// - Scalar multiplication: $\alpha\mathbf{v} = (\alpha v_1, \alpha v_2, \ldots, \alpha v_n)$
/// - Additive inverse (negation): $-\mathbf{v} = (-v_1, -v_2, \ldots, -v_n)$
///
/// ## Zero Vector Handling
///
/// This implementation represents the zero vector as an empty vector with no components.
/// Any vector with all components equal to zero is also considered a zero vector.
///
/// ## Examples
///
/// ```
/// use harness_algebra::{prelude::*, tensors::dynamic::vector::DynamicVector};
///
/// // Create a vector with components [1, 2, 3]
/// let vec1 = DynamicVector::<f64>::from([1.0, 2.0, 3.0]);
///
/// // Create the zero vector
/// let zero = DynamicVector::<f64>::zero();
///
/// // Vector addition
/// let vec2 = DynamicVector::<f64>::from([4.0, 5.0, 6.0]);
/// let sum = vec1.clone() + vec2;
///
/// // Scalar multiplication
/// let scaled = vec1 * 2.0;
/// ```
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicVector<F> {
  /// The components of the vector.
  pub components: Vec<F>,
}

impl<F> DynamicVector<F> {
  /// Creates a new `DynamicVector` from a vector of components.
  ///
  /// # Arguments
  ///
  /// * `components` - A vector containing the components of the dynamic vector
  ///
  /// # Returns
  ///
  /// A new `DynamicVector` instance with the given components
  pub const fn new(components: Vec<F>) -> Self { Self { components } }

  /// Returns the dimension (number of components) of the vector.
  ///
  /// For the zero vector (empty components), the dimension is 0.
  pub const fn dimension(&self) -> usize { self.components.len() }

  /// Returns a reference to the components of the vector.
  ///
  /// This allows read-only access to the underlying data.
  pub fn components(&self) -> &[F] { &self.components }

  /// Returns a mutable reference to the components of the vector.
  ///
  /// This allows direct modification of the underlying data.
  pub const fn components_mut(&mut self) -> &mut Vec<F> { &mut self.components }

  /// Gets a reference to the component at the specified index.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the component to retrieve
  ///
  /// # Returns
  ///
  /// A reference to the component at the specified index
  ///
  /// # Panics
  ///
  /// Panics if the index is out of bounds.
  pub fn get_component(&self, index: usize) -> &F { &self.components[index] }

  /// Sets the component at the specified index to the given value.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the component to set
  /// * `value` - The value to set at the specified index
  ///
  /// # Panics
  ///
  /// Panics if the index is out of bounds.
  pub fn set_component(&mut self, index: usize, value: F) { self.components[index] = value }

  /// Appends a value to the end of the vector, increasing its dimension by 1.
  ///
  /// # Arguments
  ///
  /// * `value` - The value to append to the vector
  pub fn append(&mut self, value: F) { self.components.push(value) }

  /// Removes the last component from the vector and returns it.
  ///
  /// # Returns
  ///
  /// The last component of the vector wrapped in `Some`, or `None` if the vector is empty.
  pub fn pop(&mut self) -> Option<F> { self.components.pop() }

  /// Returns a vector of zeros with the same dimension as the vector.
  pub fn zeros(dimension: usize) -> Self
  where F: Zero + Copy {
    Self { components: vec![F::zero(); dimension] }
  }
}

impl<F: Field> From<Vec<F>> for DynamicVector<F> {
  /// Creates a new `DynamicVector` from a `Vec<F>`.
  ///
  /// # Arguments
  ///
  /// * `components` - A vector containing the components
  fn from(components: Vec<F>) -> Self { Self { components } }
}

impl<const M: usize, F: Field + Copy> From<[F; M]> for DynamicVector<F> {
  /// Creates a new `DynamicVector` from a fixed-size array of components.
  ///
  /// # Arguments
  ///
  /// * `components` - An array of components
  fn from(components: [F; M]) -> Self { Self { components: components.to_vec() } }
}

impl<F: Field + Clone> From<&[F]> for DynamicVector<F> {
  /// Creates a new `DynamicVector` from a slice of components.
  ///
  /// # Arguments
  ///
  /// * `components` - A slice of components
  fn from(components: &[F]) -> Self { Self { components: components.to_vec() } }
}

impl<F: Field + Copy> Category for DynamicVector<F> {
  type Morphism = DynamicDenseMatrix<F, RowMajor>;

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism { f * g }

  fn identity(a: Self) -> Self::Morphism {
    let mut mat = DynamicDenseMatrix::<F, RowMajor>::new();
    for i in 0..a.dimension() {
      let mut col = Self::from(vec![F::zero(); a.dimension()]);
      col.components[i] = F::one();
      mat.append_column(&col);
    }
    mat
  }

  fn apply(f: Self::Morphism, x: Self) -> Self { f * x }
}

// TODO (autoparallel): This does handle the zero case but this is clunky as fuck and I hate it.
impl<F: Field + Copy> Add for DynamicVector<F> {
  type Output = Self;

  /// Adds two vectors component-wise.
  ///
  /// # Arguments
  ///
  /// * `other` - The vector to add to this vector
  ///
  /// # Returns
  ///
  /// A new vector representing the sum of the two vectors
  ///
  /// # Panics
  ///
  /// Panics if the vectors have different dimensions.
  fn add(self, other: Self) -> Self::Output {
    assert_eq!(self.components.len(), other.components.len());
    let mut sum = Self { components: vec![F::zero(); self.components.len()] };
    for i in 0..self.components.len() {
      sum.components[i] = self.components[i] + other.components[i];
    }
    sum
  }
}

impl<F: Field + Copy> AddAssign for DynamicVector<F> {
  /// Adds another vector to this vector in-place.
  ///
  /// # Arguments
  ///
  /// * `rhs` - The vector to add to this vector
  ///
  /// # Panics
  ///
  /// Panics if the vectors have different dimensions.
  fn add_assign(&mut self, rhs: Self) { *self = self.clone() + rhs }
}

impl<F: Field + Copy> Neg for DynamicVector<F> {
  type Output = Self;

  /// Negates this vector, returning a vector with all components negated.
  ///
  /// # Returns
  ///
  /// A new vector representing the negation of this vector
  fn neg(self) -> Self::Output {
    let mut neg = Self { components: vec![F::zero(); self.components.len()] };
    for i in 0..self.components.len() {
      neg.components[i] = -self.components[i];
    }
    neg
  }
}

impl<F: Field + Copy> Mul<F> for DynamicVector<F> {
  type Output = Self;

  /// Multiplies this vector by a scalar.
  ///
  /// # Arguments
  ///
  /// * `scalar` - The scalar to multiply by
  ///
  /// # Returns
  ///
  /// A new vector representing the product of this vector and the scalar
  fn mul(self, scalar: F) -> Self::Output {
    let mut scalar_multiple = Self { components: vec![F::zero(); self.components.len()] };
    for i in 0..self.components.len() {
      scalar_multiple.components[i] = scalar * self.components[i];
    }
    scalar_multiple
  }
}

impl<F: Field + Copy> Sub for DynamicVector<F> {
  type Output = Self;

  /// Subtracts another vector from this vector.
  ///
  /// # Arguments
  ///
  /// * `other` - The vector to subtract from this vector
  ///
  /// # Returns
  ///
  /// A new vector representing the difference between the two vectors
  ///
  /// # Panics
  ///
  /// Panics if the vectors have different dimensions.
  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<F: Field + Copy> SubAssign for DynamicVector<F> {
  /// Subtracts another vector from this vector in-place.
  ///
  /// # Arguments
  ///
  /// * `rhs` - The vector to subtract from this vector
  ///
  /// # Panics
  ///
  /// Panics if the vectors have different dimensions.
  fn sub_assign(&mut self, rhs: Self) { *self = self.clone() - rhs }
}

impl<F: Field + Copy> Additive for DynamicVector<F> {}

impl<F: Field + Copy> Group for DynamicVector<F> {
  /// Returns the identity element for vector addition (the zero vector).
  fn identity() -> Self { Self::zero() }

  /// Returns the additive inverse of this vector.
  fn inverse(&self) -> Self { -self.clone() }
}

impl<F: Field + Copy> Zero for DynamicVector<F> {
  /// Returns the zero vector (empty vector with no components).
  fn zero() -> Self { Self { components: vec![] } }

  /// Checks if this vector is the zero vector.
  ///
  /// A vector is considered zero if either:
  /// 1. It has no components (empty vector)
  /// 2. All of its components are equal to the zero element of the field
  fn is_zero(&self) -> bool {
    self.components.iter().all(|x| *x == F::zero()) || self.components.is_empty()
  }
}

impl<F: Field + Copy> AbelianGroup for DynamicVector<F> {}

impl<F: Field + Copy + Mul<Self>> LeftModule for DynamicVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> RightModule for DynamicVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> TwoSidedModule for DynamicVector<F> {
  type Ring = F;
}

impl<F: Field + Copy + Mul<Self>> VectorSpace for DynamicVector<F> {}

impl<F> Iterator for DynamicVector<F> {
  type Item = F;

  fn next(&mut self) -> Option<Self::Item> { self.components.pop() }
}

impl<F: Field + Copy + fmt::Display> fmt::Display for DynamicVector<F> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.components.is_empty() {
      return write!(f, "( )");
    }

    if self.components.len() == 1 {
      // Single element: use simple parentheses
      return write!(f, "( {} )", self.components[0]);
    }

    // Calculate the width needed for proper alignment
    let mut max_width = 0;
    for component in &self.components {
      let component_str = format!("{component}");
      max_width = max_width.max(component_str.len());
    }

    // Multi-element: use tall parentheses
    for (i, component) in self.components.iter().enumerate() {
      if i == 0 {
        writeln!(f, "⎛ {component:>max_width$} ⎞")?;
      } else if i == self.components.len() - 1 {
        write!(f, "⎝ {component:>max_width$} ⎠")?;
      } else {
        writeln!(f, "⎜ {component:>max_width$} ⎟")?;
      }
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use fixtures::Mod7;

  use super::*;

  #[test]
  fn test_zero_vector() {
    let zero_vec: DynamicVector<Mod7> = DynamicVector::zero();
    assert!(zero_vec.is_zero());
    assert_eq!(zero_vec.components.len(), 0, "Default zero vector should have 0 components");

    let non_zero_vec = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    assert!(!non_zero_vec.is_zero());

    let zero_vec_explicit_empty: DynamicVector<Mod7> = DynamicVector::from([]);
    assert!(zero_vec_explicit_empty.is_zero());
    assert_eq!(
      zero_vec_explicit_empty.components.len(),
      0,
      "Explicit empty vector should have 0 components"
    );
  }

  #[test]
  fn test_is_zero_for_non_empty_vector_with_all_zeros() {
    let vec_all_zeros: DynamicVector<Mod7> =
      DynamicVector::from([Mod7::from(0), Mod7::from(0), Mod7::from(0)]);
    assert!(vec_all_zeros.is_zero());
  }

  #[test]
  fn test_addition_zero_vectors() {
    let vec1: DynamicVector<Mod7> = DynamicVector::zero();
    let vec2: DynamicVector<Mod7> = DynamicVector::zero();
    let sum = vec1 + vec2;
    assert!(sum.is_zero());
    assert_eq!(
      sum.components.len(),
      0,
      "Sum of two default zero vectors should be a zero vector with 0 components"
    );
  }

  #[test]
  fn test_addition_same_dimension() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1)]);
    let sum = vec1 + vec2;
    assert_eq!(sum.components, vec![Mod7::from(2), Mod7::from(1)]);
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 0")]
  fn test_addition_with_zero_vector_implicit_panics() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let zero_vec: DynamicVector<Mod7> = DynamicVector::zero();
    let _ = vec1 + zero_vec; // Panics because vec1.len (2) != zero_vec.len (0)
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 3")]
  fn test_addition_different_dimensions_panic() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1), Mod7::from(1)]);
    let _ = vec1 + vec2; // Should panic
  }

  #[test]
  fn test_negation() {
    let vec = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let neg_vec = -vec;
    assert_eq!(neg_vec.components, vec![Mod7::from(6), Mod7::from(0)]);
  }

  #[test]
  fn test_scalar_multiplication() {
    let vec = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0), Mod7::from(1)]);
    let scalar_one = Mod7::from(1);
    let scalar_zero = Mod7::from(0);

    let mul_one = vec.clone() * scalar_one;
    assert_eq!(mul_one.components, vec![
      Mod7::from(1) * scalar_one,
      Mod7::from(0) * scalar_one,
      Mod7::from(1) * scalar_one
    ]);

    let mul_zero = vec * scalar_zero;
    assert_eq!(mul_zero.components, vec![Mod7::from(0), Mod7::from(0), Mod7::from(0)]);
  }

  #[test]
  fn test_subtraction_same_dimension() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(0), Mod7::from(1)]);
    let diff = vec1 + (-vec2);
    assert_eq!(diff.components, vec![Mod7::from(1), Mod7::from(0)]);
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 1")]
  fn test_subtraction_different_dimensions_panic() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1)]);
    let _ = vec1 - vec2;
  }

  #[test]
  fn test_add_assign_same_dimension() {
    let mut vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1)]);
    vec1 += vec2;
    assert_eq!(vec1.components, vec![Mod7::from(2), Mod7::from(1)]);
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 1")]
  fn test_add_assign_different_dimensions_panic() {
    let mut vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1)]);
    vec1 += vec2; // Should panic
  }

  #[test]
  fn test_sub_assign_same_dimension() {
    let mut vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(0), Mod7::from(1)]);
    vec1 -= vec2;
    assert_eq!(vec1.components, vec![Mod7::from(1), Mod7::from(0)]);
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 1")]
  fn test_sub_assign_different_dimensions_panic() {
    let mut vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1)]);
    vec1 -= vec2; // Should panic
  }

  #[test]
  fn test_zeros() {
    let zero_vec = DynamicVector::<Mod7>::zeros(3);
    assert_eq!(zero_vec.components, vec![Mod7::from(0), Mod7::from(0), Mod7::from(0)]);
  }

  #[test]
  fn test_display_formatting() {
    // Test empty vector
    let empty: DynamicVector<f64> = DynamicVector::new(vec![]);
    println!("Empty vector: \n{empty}");

    // Test single element
    let single = DynamicVector::from([42.0]);
    println!("Single element: \n{single}");

    // Test multiple elements
    let multi = DynamicVector::from([1.0, 2.5, -3.7, 0.0]);
    println!("Multiple elements: \n{multi}");
  }
}
