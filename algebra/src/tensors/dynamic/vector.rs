// TODO (autoparallel): We could use `MaybeUninit` to avoid the `Vec` allocation especially in the
// zero case.

use super::*;

/// A dynamically-sized vector (typically with components from a field `F`).
///
/// The dimension can be determined at runtime, making it flexible for various applications.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicVector<F> {
  components: Vec<F>,
}

impl<F> DynamicVector<F> {
  pub fn new(components: Vec<F>) -> Self { Self { components } }

  pub fn dimension(&self) -> usize { self.components.len() }

  pub fn components(&self) -> &[F] { &self.components }

  pub fn components_mut(&mut self) -> &mut Vec<F> { &mut self.components }

  pub fn get_component(&self, index: usize) -> &F { &self.components[index] }

  pub fn set_component(&mut self, index: usize, value: F) { self.components[index] = value }

  pub fn append(&mut self, value: F) { self.components.push(value) }

  pub fn pop(&mut self) -> Option<F> { self.components.pop() }
}

impl<F: Field> From<Vec<F>> for DynamicVector<F> {
  fn from(components: Vec<F>) -> Self { Self { components } }
}

impl<const M: usize, F: Field + Copy> From<[F; M]> for DynamicVector<F> {
  fn from(components: [F; M]) -> Self { Self { components: components.to_vec() } }
}

impl<F: Field + Clone> From<&[F]> for DynamicVector<F> {
  fn from(components: &[F]) -> Self { Self { components: components.to_vec() } }
}

// TODO: This does handle the zero case but this is clunky as fuck and I hate it.
impl<F: Field + Copy> Add for DynamicVector<F> {
  type Output = Self;

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
  fn add_assign(&mut self, rhs: Self) { *self = self.clone() + rhs }
}

impl<F: Field + Copy> Neg for DynamicVector<F> {
  type Output = Self;

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

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<F: Field + Copy> SubAssign for DynamicVector<F> {
  fn sub_assign(&mut self, rhs: Self) { *self = self.clone() - rhs }
}

impl<F: Field + Copy> Additive for DynamicVector<F> {}

impl<F: Field + Copy> Group for DynamicVector<F> {
  fn identity() -> Self { Self::zero() }

  fn inverse(&self) -> Self { -self.clone() }
}

impl<F: Field + Copy> Zero for DynamicVector<F> {
  fn zero() -> Self { Self { components: vec![] } }

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
    let _sum = vec1 + zero_vec; // Panics because vec1.len (2) != zero_vec.len (0)
  }

  #[test]
  #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 3")]
  fn test_addition_different_dimensions_panic() {
    let vec1 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(0)]);
    let vec2 = DynamicVector::<Mod7>::from([Mod7::from(1), Mod7::from(1), Mod7::from(1)]);
    let _sum = vec1 + vec2; // Should panic
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
    let _diff = vec1 - vec2;
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
}
