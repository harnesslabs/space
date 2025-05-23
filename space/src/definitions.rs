//! # Core Definitions for Mathematical Spaces
//!
//! This module lays the groundwork for representing various mathematical spaces
//! by defining a set of core traits. These traits abstract fundamental properties
//! and operations associated with topological spaces, metric spaces, normed spaces,
//! and inner product spaces. The hierarchy of these traits reflects the way these
//! mathematical structures build upon one another.
//!
//! ## Trait Hierarchy and Concepts
//!
//! 1. **[`Collection`]**: (Defined in `crate::set`) At the base, any space is a collection of items
//!    (points). This trait provides basic operations like checking for containment and emptiness.
//!
//! 2. **[`Topology`]**: Extends [`Collection`]. A topological space is a set of points endowed with
//!    a structure, called a topology, which allows defining concepts such as continuity,
//!    connectedness, and convergence. This trait focuses on operations like finding neighborhoods
//!    and computing boundaries of items within the space, crucial for algebraic topology operations
//!    like homology (which uses [`Chain`]).
//!
//! 3. **[`MetricSpace`]**: Extends [`Collection`]. A metric space formalizes the concept of
//!    distance between points. It introduces a `distance` function. Every metric space can induce a
//!    topology (the metric topology), where open sets are defined using open balls.
//!
//! 4. **[`NormedSpace`]**: Extends [`MetricSpace`]. A normed space is a vector space where each
//!    vector is assigned a length or "norm." The norm induces a metric, making every normed space a
//!    metric space ($d(x,y) = ||x-y||$).
//!
//! 5. **[`InnerProductSpace`]**: Extends [`NormedSpace`]. An inner product space is a vector space
//!    equipped with an inner product (or scalar product), which allows defining geometric notions
//!    like angles and orthogonality. The inner product induces a norm ($||x|| = \sqrt{\langle x, x
//!    \rangle}$), making every inner product space a normed space.
//!
//! ## Usage
//!
//! These traits are intended to be implemented by specific data structures that model these
//! mathematical spaces. For example, a struct representing a simplicial complex might implement
//! `Topology`, while a struct for Euclidean vectors might implement `InnerProductSpace`.
//! By programming against these traits, algorithms in geometry, topology, and related fields
//! can be written more generically.
//!
//! ## Future Considerations
//!
//! - The relationship between `MetricSpace` and `Topology` (a metric induces a topology) could be
//!   further formalized, perhaps through blanket implementations or associated functions if a
//!   `MetricSpace` is also to be treated as a `Topology` directly through its induced structure.
//! - Default implementations for relationships, e.g., a `MetricSpace` deriving its `Topology` from
//!   its metric, or a `NormedSpace` deriving its `distance` function from its `norm`.

use harness_algebra::rings::{Field, Ring};

use crate::{homology::Chain, set::Collection};

/// Defines the properties and operations of a topological space.
///
/// A topological space consists of a set of points (items) along with a topology,
/// which is a set of open sets satisfying certain axioms. This trait abstracts
/// operations on such spaces, particularly those relevant for constructing
/// chain complexes and computing homology.
///
/// It extends [`Collection`], indicating that a topological space is fundamentally
/// a collection of items.
///
/// # Type Parameters
///
/// The type implementing `Topology` is `Self`, representing the specific topological space.
/// `Self::Item` is the type of points or fundamental components (e.g., cells, simplices)
/// within the space.
pub trait Topology: Sized + Collection {
  /// Returns the neighborhood of a given item in the topological space.
  ///
  /// The definition of a "neighborhood" can vary depending on the specific type of
  /// topological space (e.g., for a point, it might be a set of open sets containing it;
  /// for a cell in a complex, it might be adjacent cells or cofaces).
  /// The exact semantics should be defined by the implementing type.
  ///
  /// # Arguments
  ///
  /// * `item`: A reference to an item in the space for which the neighborhood is to be computed.
  ///
  /// # Returns
  ///
  /// A `Vec<Self::Item>` containing the items that form the neighborhood of the input `item`.
  fn neighborhood(&self, item: &Self::Item) -> Vec<Self::Item>;

  /// Computes the boundary of a given item in the topological space, represented as a formal chain.
  ///
  /// The boundary operation, denoted $\partial$, is a fundamental concept in algebraic topology.
  /// For a $k$-item (e.g., a $k$-simplex or $k$-cell), its boundary is a formal sum of
  /// $(k-1)$-items with coefficients in a ring `R`. This method provides this boundary as a
  /// [`Chain`].
  ///
  /// For example, the boundary of an edge (1-simplex) $[v_0, v_1]$ is $v_1 - v_0$.
  /// The boundary of a triangle (2-simplex) $[v_0, v_1, v_2]$ is $[v_1, v_2] - [v_0, v_2] + [v_0,
  /// v_1]$.
  ///
  /// # Type Parameters
  ///
  /// * `R`: The type of coefficients for the resulting boundary chain. It must implement [`Ring`]
  ///   and be `Copy`.
  ///
  /// # Arguments
  ///
  /// * `item`: A reference to an item in the space whose boundary is to be computed.
  ///
  /// # Returns
  ///
  /// A [`Chain<'_, Self, R>`] representing the formal sum of items that constitute the boundary
  /// of the input `item`, with coefficients from the ring `R`.
  fn boundary<R: Ring + Copy>(&self, item: &Self::Item) -> Chain<'_, Self, R>;
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
    point_a: <Self as Collection>::Item,
    point_b: <Self as Collection>::Item,
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
  fn norm(point: Self::Item) -> Self::Norm;
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
  fn inner_product(&self, point_a: Self::Item, point_b: Self::Item) -> Self::InnerProduct;
}
