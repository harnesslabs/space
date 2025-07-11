//! # Cloud - Vector Set in N-dimensional Space
//!
//! This module defines the [`Cloud`] struct and its implementations, representing a collection
//! of points (vectors) in an `N`-dimensional space.
//!
//! ## Overview
//!
//! A [`Cloud`] is essentially a set of N-dimensional vectors with elements [`f64`].
//! The module provides:
//!
//! - Basic collection operations ([`Collection::contains`], [`Collection::is_empty`])
//! - Metric space capabilities ([`MetricSpace::distance`])
//! - Normed space functionality ([`NormedSpace::norm`])
//!
//! ## Example
//!
//! ```
//! use cova_algebra::tensors::SVector;
//! use cova_space::{cloud::Cloud, prelude::*};
//!
//! // Create two 2D vectors
//! let v1 = SVector::from([1.0, 2.0]);
//! let v2 = SVector::from([3.0, 4.0]);
//!
//! // Create a cloud containing these vectors
//! let cloud = Cloud::new(vec![v1, v2]);
//!
//! // Check if the cloud contains a vector
//! assert!(cloud.contains(&v1));
//!
//! // Calculate distance between vectors
//! let distance = Cloud::<2>::distance(v1, v2);
//! ```
//!
//! ## Implementation Details
//!
//! The [`Cloud`] implements several traits:
//! - [`Collection`] - Basic set operations
//! - [`MetricSpace`] - Distance calculations
//! - [`NormedSpace`] - Norm calculations (Euclidean norm)

use cova_algebra::tensors::SVector;

use crate::{
  definitions::{MetricSpace, NormedSpace},
  set::Collection,
};

/// Defines the [`Cloud`] struct, representing a collection of points (vectors)
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
pub struct Cloud<const N: usize> {
  points: Vec<SVector<f64, N>>,
}

impl<const N: usize> Cloud<N> {
  /// Creates a new `Cloud` from a given set of points.
  ///
  /// # Arguments
  ///
  /// * `points`: A `HashSet` of `Vector<N, F>` representing the points in the cloud.
  pub const fn new(points: Vec<SVector<f64, N>>) -> Self { Self { points } }

  /// Returns a reference to the points in the cloud.
  pub const fn points_ref(&self) -> &Vec<SVector<f64, N>> { &self.points }
}

impl<const N: usize> Collection for Cloud<N> {
  type Item = SVector<f64, N>;

  fn contains(&self, point: &Self::Item) -> bool { self.points.contains(point) }

  fn is_empty(&self) -> bool { self.points.is_empty() }
}

impl<const N: usize> MetricSpace for Cloud<N> {
  type Distance = f64;

  /// Calculates the distance between two points in the cloud.
  ///
  /// The distance is defined as the norm of the difference between the two points.
  fn distance(point_a: Self::Item, point_b: Self::Item) -> Self::Distance {
    <Self as NormedSpace>::norm(point_a - point_b)
  }
}

impl<const N: usize> NormedSpace for Cloud<N> {
  type Norm = f64;

  /// Calculates the norm of a point.
  ///
  /// The norm is defined as the sum of the squares of its components (Euclidean norm).
  fn norm(point: Self::Item) -> Self::Norm { point.iter().map(|p| *p * *p).sum::<f64>().sqrt() }
}

#[cfg(test)]
mod tests {
  #![allow(clippy::float_cmp)]

  use super::*;

  fn create_test_vector1() -> SVector<f64, 2> { SVector::from([1.0, 2.0]) }

  fn create_test_vector2() -> SVector<f64, 2> { SVector::from([3.0, 4.0]) }

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
    let points: Vec<SVector<f64, 2>> = Vec::new();
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
    assert_eq!(Cloud::<2>::norm(v1), 5.0_f64.sqrt());

    let v2 = create_test_vector2(); // [3.0, 4.0]
                                    // 3.0*3.0 + 4.0*4.0 = 9.0 + 16.0 = 25.0
    assert_eq!(Cloud::<2>::norm(v2), 25.0_f64.sqrt());
  }

  #[test]
  fn test_distance() {
    let v1 = create_test_vector1(); // [1.0, 2.0]
    let v2 = create_test_vector2(); // [3.0, 4.0]
                                    // v1 - v2 = [-2.0, -2.0]
                                    // norm([-2.0, -2.0]) = (-2.0)*(-2.0) + (-2.0)*(-2.0) = 4.0 + 4.0 = 8.0
    assert_eq!(Cloud::<2>::distance(v1, v2), 8.0_f64.sqrt());
  }
}
