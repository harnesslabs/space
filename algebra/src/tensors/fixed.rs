use super::*;

/// A fixed-size vector over a field.
///
/// This is a concrete implementation of a vector space, where vectors have
/// a fixed number of components and the scalars come from a field.
///
/// ```
/// use harness_algebra::{rings::Field, tensors::fixed::FixedVector};
///
/// let v = FixedVector::<3, f64>([1.0, 2.0, 3.0]);
/// let w = FixedVector::<3, f64>([4.0, 5.0, 6.0]);
/// let sum = v + w;
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedVector<const M: usize, F: Field>(pub [F; M]);

impl<const M: usize, F: Field + Copy> Default for FixedVector<M, F> {
  fn default() -> Self { Self([F::zero(); M]) }
}

impl<const M: usize, F: Field + Copy> Add for FixedVector<M, F> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    let mut sum = Self::zero();
    for i in 0..M {
      sum.0[i] = self.0[i] + other.0[i];
    }
    sum
  }
}

impl<const M: usize, F: Field + Copy> AddAssign for FixedVector<M, F> {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs }
}

impl<const M: usize, F: Field + Copy> Neg for FixedVector<M, F> {
  type Output = Self;

  fn neg(self) -> Self::Output {
    let mut neg = Self::zero();
    for i in 0..M {
      neg.0[i] = -self.0[i];
    }
    neg
  }
}

impl<const M: usize, F: Field + Copy> Mul<F> for FixedVector<M, F> {
  type Output = Self;

  fn mul(self, scalar: F) -> Self::Output {
    let mut scalar_multiple = Self::zero();
    for i in 0..M {
      scalar_multiple.0[i] = scalar * self.0[i];
    }
    scalar_multiple
  }
}

impl<const M: usize, F: Field + Copy> Sub for FixedVector<M, F> {
  type Output = Self;

  fn sub(self, other: Self) -> Self::Output { self + -other }
}

impl<const M: usize, F: Field + Copy> SubAssign for FixedVector<M, F> {
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs }
}

impl<const M: usize, F: Field + Copy> Additive for FixedVector<M, F> {}

impl<const M: usize, F: Field + Copy> Group for FixedVector<M, F> {
  fn identity() -> Self { Self::zero() }

  fn inverse(&self) -> Self { -*self }
}

impl<const M: usize, F: Field + Copy> Zero for FixedVector<M, F> {
  fn zero() -> Self { Self([F::zero(); M]) }

  fn is_zero(&self) -> bool { self.0.iter().all(|x| *x == F::zero()) }
}

impl<const M: usize, F: Field + Copy> AbelianGroup for FixedVector<M, F> {}

impl<const M: usize, F: Field + Copy + Mul<Self>> LeftModule for FixedVector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> RightModule for FixedVector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> TwoSidedModule for FixedVector<M, F> {
  type Ring = F;
}

impl<const M: usize, F: Field + Copy + Mul<Self>> VectorSpace for FixedVector<M, F> {}

impl<const M: usize, F: Field> From<[F; M]> for FixedVector<M, F> {
  fn from(components: [F; M]) -> Self { Self(components) }
}

impl<const M: usize, F: Field + Copy> From<&[F; M]> for FixedVector<M, F> {
  fn from(components: &[F; M]) -> Self { Self(*components) }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::fixtures::Mod7;

  #[test]
  fn test_zero_vector() {
    let zero_vec: FixedVector<3, Mod7> = FixedVector::zero();
    assert!(zero_vec.is_zero());
    assert_eq!(zero_vec.0, [Mod7::zero(), Mod7::zero(), Mod7::zero()]);

    let zero_vec_default: FixedVector<3, Mod7> = FixedVector::default();
    assert!(zero_vec_default.is_zero());
    assert_eq!(zero_vec_default.0, [Mod7::zero(), Mod7::zero(), Mod7::zero()]);

    let non_zero_vec = FixedVector::from([Mod7::from(1), Mod7::zero(), Mod7::from(2)]);
    assert!(!non_zero_vec.is_zero());
  }

  #[test]
  fn test_is_zero_all_components_zero() {
    let vec: FixedVector<2, Mod7> = FixedVector::from([Mod7::zero(), Mod7::zero()]);
    assert!(vec.is_zero());
  }

  #[test]
  fn test_from_array() {
    let arr = [Mod7::from(1), Mod7::from(2), Mod7::from(3)];
    let vec: FixedVector<3, Mod7> = FixedVector::from(arr);
    assert_eq!(vec.0, arr);
  }

  #[test]
  fn test_addition() {
    let vec1 = FixedVector::from([Mod7::from(1), Mod7::from(2)]);
    let vec2 = FixedVector::from([Mod7::from(3), Mod7::from(4)]);
    let sum = vec1 + vec2;
    assert_eq!(sum.0, [Mod7::from(4), Mod7::from(6)]);
  }

  #[test]
  fn test_add_assign() {
    let mut vec1 = FixedVector::from([Mod7::from(1), Mod7::from(2)]);
    let vec2 = FixedVector::from([Mod7::from(3), Mod7::from(4)]);
    vec1 += vec2;
    assert_eq!(vec1.0, [Mod7::from(4), Mod7::from(6)]);
  }

  #[test]
  fn test_negation() {
    let vec = FixedVector::from([Mod7::from(1), Mod7::from(0), Mod7::from(6)]);
    let neg_vec = -vec;
    assert_eq!(neg_vec.0, [Mod7::from(6), Mod7::from(0), Mod7::from(1)]);
  }

  #[test]
  fn test_scalar_multiplication() {
    let vec = FixedVector::from([Mod7::from(1), Mod7::from(2), Mod7::from(3)]);
    let scalar = Mod7::from(2);
    let product = vec * scalar;
    assert_eq!(product.0, [Mod7::from(2), Mod7::from(4), Mod7::from(6)]);

    let scalar_zero = Mod7::zero();
    let product_zero = vec * scalar_zero;
    assert_eq!(product_zero.0, [Mod7::zero(), Mod7::zero(), Mod7::zero()]);
  }

  #[test]
  fn test_subtraction() {
    let vec1 = FixedVector::from([Mod7::from(5), Mod7::from(3)]);
    let vec2 = FixedVector::from([Mod7::from(1), Mod7::from(4)]);
    let diff = vec1 - vec2;
    // vec1 + (-vec2) = [5,3] + [-1,-4] = [5,3] + [6,3] = [11,6] = [4,6]
    assert_eq!(diff.0, [Mod7::from(4), Mod7::from(6)]);
  }

  #[test]
  fn test_sub_assign() {
    let mut vec1 = FixedVector::from([Mod7::from(5), Mod7::from(3)]);
    let vec2 = FixedVector::from([Mod7::from(1), Mod7::from(4)]);
    vec1 -= vec2;
    assert_eq!(vec1.0, [Mod7::from(4), Mod7::from(6)]);
  }
}
