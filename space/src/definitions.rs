//! Core mathematical definitions for topological and metric spaces.
//!
//! This module provides trait definitions for fundamental mathematical structures
//! used in topology and geometry. The traits form a hierarchy from basic set operations
//! up through inner product spaces.

use crate::set::{Collection, Set};

/// A trait for topological spaces.
///
/// A topological space consists of a set of points together with a collection of open sets
/// that satisfy certain axioms. This trait provides methods for working with neighborhoods
/// and testing if sets are open.
///
/// # Type Parameters
/// * `Point` - The type of points in the space
/// * `OpenSet` - The type representing open sets in the space
pub trait TopologicalSpace {
  /// The type of points in the space
  type Point;
  /// The type representing open sets in the space
  type OpenSet: Set<Point = Self::Point>;

  /// Returns a neighborhood of a given point.
  ///
  /// In topology, a neighborhood of a point is an open set containing that point.
  ///
  /// # Arguments
  /// * `point` - The point whose neighborhood to compute
  fn neighborhood(&self, point: Self::Point) -> Self::OpenSet;

  /// Tests if a given set is open in this topological space.
  ///
  /// # Arguments
  /// * `open_set` - The set to test for openness
  fn is_open(&self, open_set: Self::OpenSet) -> bool;
}

// TODO: If a metric space is also a normed space, then it should implement both traits and have a
// default way to compute the distance between two points using the norm.
// TODO: Previously I had a trait for metric spaces that extended `TopologicalSpace`, but this
// was problematic because you really get a topology from the metric.
/// A trait for metric spaces.
///
/// A metric space is a set together with a notion of distance between points.
/// This trait extends `TopologicalSpace` by adding a distance function.
///
/// # Type Parameters
/// * `Distance` - The type used to represent distances between points
pub trait MetricSpace: Collection {
  /// The type used to represent distances between points
  type Distance;

  /// Computes the distance between two points.
  ///
  /// # Arguments
  /// * `point_a` - The first point
  /// * `point_b` - The second point
  fn distance(
    point_a: <Self as Collection>::Point,
    point_b: <Self as Collection>::Point,
  ) -> Self::Distance;
}

/// A trait for normed spaces.
///
/// A normed space is a vector space equipped with a norm function that assigns
/// a size to each vector. This trait extends `MetricSpace` by adding a norm function.
///
/// # Type Parameters
/// * `Norm` - The type used to represent the norm of a vector
pub trait NormedSpace: MetricSpace {
  /// The type used to represent the norm of a vector
  type Norm;

  /// Computes the norm (length/magnitude) of a point/vector.
  ///
  /// # Arguments
  /// * `point` - The point whose norm to compute
  fn norm(point: Self::Point) -> Self::Norm;
}

/// A trait for inner product spaces.
///
/// An inner product space is a vector space equipped with an inner product operation
/// that allows for notions of angle and orthogonality. This trait extends `NormedSpace`
/// by adding an inner product function.
///
/// # Type Parameters
/// * `InnerProduct` - The type used to represent inner products
pub trait InnerProductSpace: NormedSpace {
  /// The type used to represent inner products
  type InnerProduct;

  /// Computes the inner product of two points/vectors.
  ///
  /// # Arguments
  /// * `point_a` - The first point
  /// * `point_b` - The second point
  fn inner_product(&self, point_a: Self::Point, point_b: Self::Point) -> Self::InnerProduct;
}
