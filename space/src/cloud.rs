//! # Cloud - Vector Set in N-dimensional Space
//!
//! This module defines the `Cloud` struct and its implementations, representing a collection
//! of points (vectors) in an `N-`dimensional space.
//!
//! ## Overview
//!
//! A [`Cloud<N, F>`] is essentially a set of N-dimensional vectors with elements from a field `F`.
//! The module provides:
//!
//! - Basic collection operations ([`Collection::contains`], [`Collection::is_empty`])
//! - Metric space capabilities ([`MetricSpace::distance`])
//! - Normed space functionality ([`NormedSpace::norm`])
//!
//! ## Example
//!
//! ```
//! use harness_algebra::tensors::fixed::FixedVector;
//! use harness_space::{cloud::Cloud, prelude::*};
//!
//! // Create two 2D vectors
//! let v1 = FixedVector([1.0, 2.0]);
//! let v2 = FixedVector([3.0, 4.0]);
//!
//! // Create a cloud containing these vectors
//! let cloud = Cloud::new(vec![v1, v2]);
//!
//! // Check if the cloud contains a vector
//! assert!(cloud.contains(&v1));
//!
//! // Calculate distance between vectors
//! let distance = Cloud::<2, f64>::distance(v1, v2);
//! ```
//!
//! ## Implementation Details
//!
//! The `Cloud` implements several traits:
//! - `Collection` - Basic set operations
//! - `MetricSpace` - Distance calculations
//! - `NormedSpace` - Norm calculations (Euclidean norm)

use std::iter::Sum;

use harness_algebra::{rings::Field, tensors::fixed::FixedVector};

use crate::{
  definitions::{MetricSpace, NormedSpace},
  set::Collection,
};

/// Defines the `Cloud` struct, representing a collection of points (vectors)
/// in an N-dimensional space.
///
/// This module provides the `Cloud` type, which can be used to store and
/// manage a set of points. It implements traits for basic collection operations,
/// as well as for metric and normed space concepts, allowing for calculations
/// like distance and norm.
///
/// A `Cloud` is essentially a set of vectors, providing basic [`Collection`] operations
/// as well as metric and normed space functionalities.
#[derive(Debug, Clone)]
pub struct Cloud<const N: usize, F: Field> {
  points: Vec<FixedVector<N, F>>,
}

impl<F: Field, const N: usize> Cloud<N, F> {
  /// Creates a new `Cloud` from a given set of points.
  ///
  /// # Arguments
  ///
  /// * `points`: A `HashSet` of `Vector<N, F>` representing the points in the cloud.
  pub fn new(points: Vec<FixedVector<N, F>>) -> Self { Self { points } }

  /// Returns a reference to the points in the cloud.
  pub fn points_ref(&self) -> &Vec<FixedVector<N, F>> { &self.points }
}

// impl<F: Field + Copy, const N: usize> Set for Cloud<N, F> {
//   type Point = Vector<N, F>;

//   /// Checks if the cloud contains a given point.
//   fn contains(&self, point: &Self::Point) -> bool { self.points.contains(point) }

//   /// Returns a new cloud containing points that are in `self` but not in `other`.
//   fn minus(&self, other: &Self) -> Self { Self { points: self.points.minus(&other.points) } }

//   /// Returns a new cloud containing points that are common to both `self` and `other`.
//   fn meet(&self, other: &Self) -> Self { Self { points: self.points.meet(&other.points) } }

//   /// Returns a new cloud containing all points from `self` and `other`.
//   fn join(&self, other: &Self) -> Self { Self { points: self.points.join(&other.points) } }

//   /// Checks if the cloud is empty.
//   fn is_empty(&self) -> bool { self.points.is_empty() }
// }

impl<const N: usize, F: Field + Copy + Sum<F>> Collection for Cloud<N, F> {
  type Point = FixedVector<N, F>;

  fn contains(&self, point: &Self::Point) -> bool { self.points.contains(point) }

  fn is_empty(&self) -> bool { self.points.is_empty() }
}

impl<const N: usize, F: Field + Copy + Sum<F>> MetricSpace for Cloud<N, F> {
  type Distance = F;

  /// Calculates the distance between two points in the cloud.
  ///
  /// The distance is defined as the norm of the difference between the two points.
  fn distance(point_a: Self::Point, point_b: Self::Point) -> Self::Distance {
    <Self as NormedSpace>::norm(point_a - point_b)
  }
}

impl<const N: usize, F: Field + Copy + Sum<F>> NormedSpace for Cloud<N, F> {
  type Norm = F;

  /// Calculates the norm of a point.
  ///
  /// The norm is defined as the sum of the squares of its components (Euclidean norm).
  fn norm(point: Self::Point) -> Self::Norm { point.0.iter().map(|p| *p * *p).sum() }
}

#[cfg(test)]
mod tests {
  #![allow(clippy::float_cmp)]

  use super::*;

  fn create_test_vector1() -> FixedVector<2, f64> { FixedVector([1.0, 2.0]) }

  fn create_test_vector2() -> FixedVector<2, f64> { FixedVector([3.0, 4.0]) }

  #[test]
  fn test_new_cloud() {
    let points = vec![create_test_vector1()];
    let cloud = Cloud::new(points.clone());
    assert_eq!(cloud.points, points);
  }

  #[test]
  fn test_contains_point() {
    let points = vec![create_test_vector1()];
    let cloud = Cloud::new(points);
    assert!(cloud.contains(&create_test_vector1()));
    assert!(!cloud.contains(&create_test_vector2()));
  }

  #[test]
  fn test_is_empty() {
    let points: Vec<FixedVector<2, f64>> = Vec::new();
    let cloud = Cloud::new(points);
    assert!(cloud.is_empty());

    let points_non_empty = vec![create_test_vector1()];
    let cloud_non_empty = Cloud::new(points_non_empty);
    assert!(!cloud_non_empty.is_empty());
  }

  #[test]
  fn test_norm() {
    let v1 = create_test_vector1(); // [1.0, 2.0]
                                    // 1.0*1.0 + 2.0*2.0 = 1.0 + 4.0 = 5.0
    assert_eq!(Cloud::<2, f64>::norm(v1), 5.0);

    let v2 = create_test_vector2(); // [3.0, 4.0]
                                    // 3.0*3.0 + 4.0*4.0 = 9.0 + 16.0 = 25.0
    assert_eq!(Cloud::<2, f64>::norm(v2), 25.0);
  }

  #[test]
  fn test_distance() {
    let v1 = create_test_vector1(); // [1.0, 2.0]
    let v2 = create_test_vector2(); // [3.0, 4.0]
                                    // v1 - v2 = [-2.0, -2.0]
                                    // norm([-2.0, -2.0]) = (-2.0)*(-2.0) + (-2.0)*(-2.0) = 4.0 + 4.0 = 8.0
    assert_eq!(Cloud::<2, f64>::distance(v1, v2), 8.0);
  }
}
