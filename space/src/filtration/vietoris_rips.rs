// space/src/filtration/vietoris_rips.rs
//! # Vietoris-Rips Complex Construction
//!
//! This module implements the construction of Vietoris-Rips complexes, a fundamental tool
//! in topological data analysis for approximating the shape of a point cloud.
//!
//! ## Definition
//!
//! Given a finite set of points $X = \\{x_1, x_2, \dots, x_n\\}$ in a metric space $(M, d)$
//! and a real number $\\epsilon > 0$ (the distance threshold), the **Vietoris-Rips complex**
//! $VR_\\epsilon(X)$ is an abstract simplicial complex whose vertices are the points in $X$.
//! A $k$-simplex $[x_{i_0}, x_{i_1}, \dots, x_{i_k}]$ is included in $VR_\\epsilon(X)$ if and only
//! if the distance between any pair of its vertices is less than or equal to $\\epsilon$. That is:
//!
//! \\[
//! [x_{i_0}, x_{i_1}, \dots, x_{i_k}] \\in VR_\\epsilon(X) \\iff d(x_{i_j}, x_{i_l}) \\le \\epsilon
//! \\quad \\forall j, l \\in \\{0, 1, \dots, k\\} \\]
//!
//! In simpler terms, a set of points forms a simplex if all points in that set are pairwise
//! within distance $\\epsilon$ of each other.
//!
//! ## Filtration
//!
//! The Vietoris-Rips complex naturally forms a filtration. As $\\epsilon$ increases, more simplices
//! are added to the complex, and no simplices are ever removed:
//!
//! \\[
//! VR_{\\epsilon_1}(X) \\subseteq VR_{\\epsilon_2}(X) \\quad \\text{if} \\quad \\epsilon_1 \\le
//! \\epsilon_2 \\]
//!
//! This property is crucial for persistent homology, which studies how the homology groups
//! of the complex change as $\\epsilon$ varies.
//!
//! ## Usage
//!
//! This module provides the [`VietorisRips`] struct, which implements the
//! [`Filtration`] trait. It takes a [`Cloud`] of points and an `epsilon` value to produce a
//! [`SimplicialComplex`] or, optionally, [`HomologyGroup`]s for specified dimensions.
//!
//! ```rust
//! use harness_algebra::tensors::fixed::FixedVector;
//! use harness_space::{
//!   cloud::Cloud,
//!   filtration::{vietoris_rips::VietorisRips, Filtration},
//!   simplicial::SimplicialComplex,
//! };
//!
//! // Example: Create a cloud of 3 points forming a triangle
//! let p0 = FixedVector([0.0, 0.0]);
//! let p1 = FixedVector([1.0, 0.0]);
//! let p2 = FixedVector([0.5, 0.866]); // Approx. equilateral triangle
//! let cloud: Cloud<2, f64> = Cloud::new(vec![p0, p1, p2]);
//!
//! // Create a VietorisRips builder for SimplicialComplex output
//! let vr_builder = VietorisRips::<2, f64, SimplicialComplex>::new();
//!
//! // Build the complex with epsilon = 1.1 (all points are within this distance)
//! let complex = vr_builder.build(&cloud, 1.1, &());
//!
//! assert_eq!(complex.simplices_by_dimension(0).unwrap().len(), 3); // 3 vertices
//! assert_eq!(complex.simplices_by_dimension(1).unwrap().len(), 3); // 3 edges
//! assert_eq!(complex.simplices_by_dimension(2).unwrap().len(), 1); // 1 triangle (2-simplex)
//! ```
//!
//! When the `"parallel"` feature is enabled, this module also provides implementations for
//! [`ParallelFiltration`] to leverage multi-core processing
//! for building the filtration and computing homology.

use std::{
  collections::{HashMap, HashSet},
  iter::Sum,
  marker::PhantomData,
};

use harness_algebra::rings::Field;
use itertools::Itertools;

#[cfg(feature = "parallel")]
use crate::filtration::ParallelFiltration;
use crate::{
  cloud::Cloud,
  complexes::simplicial::{HomologyGroup, Simplex, SimplicialComplex},
  filtration::Filtration,
  prelude::MetricSpace,
};

/// A struct that allows construction of Vietoris-Rips complexes.
///
/// It implements the [`Filtration`] trait, taking a [`Cloud`] of points and a distance
/// threshold `epsilon` to produce a [`SimplicialComplex`] or [`HomologyGroup`]s.
///
/// A $k$-simplex `[v0, v1, ..., vk]` is included in the Vietoris-Rips complex if
/// the distance between any pair of its vertices `(vi, vj)` is less than or equal
/// to the threshold `epsilon`.
///
/// # Type Parameters
///
/// * `N`: The dimension of the Euclidean space where the points reside.
/// * `F`: The numeric type for coordinates and distances (must be a [`Field`]).
/// * `O`: The output type of the filtration. This is typically [`SimplicialComplex`] or
///   [`HashMap<usize, HomologyGroup<R>>`](HashMap) if computing homology directly.
pub struct VietorisRips<const N: usize, F, O> {
  _phantom:      PhantomData<[F; N]>, // To use N and F generics
  _output_space: PhantomData<O>,      // To specialize filtration output
}

impl<const N: usize, F, O> VietorisRips<N, F, O> {
  /// Creates a new `VietorisRips` builder.
  ///
  /// The specific behavior of the builder (e.g., what it produces) is determined
  /// by the trait implementations for `Filtration` based on the type parameter `O`.
  pub fn new() -> Self { Self { _phantom: PhantomData, _output_space: PhantomData } }
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> VietorisRips<N, F, SimplicialComplex> {
  /// Builds a Vietoris-Rips [`SimplicialComplex`] from a given point [`Cloud`] and distance
  /// threshold `epsilon`.
  ///
  /// # Arguments
  ///
  /// * `cloud`: A reference to a [`Cloud<N, F>`] containing the input points.
  /// * `epsilon`: The distance threshold $F$. A simplex is formed by a set of points if all
  ///   pairwise distances within that set are less than or equal to `epsilon`.
  ///
  /// # Returns
  ///
  /// A [`SimplicialComplex`] representing the Vietoris-Rips complex for the given
  /// `cloud` and `epsilon`.
  ///
  /// # Details
  ///
  /// 1. All points in the `cloud` are added as 0-simplices (vertices).
  /// 2. For $k > 0$, a $k$-simplex is formed by $k+1$ vertices if all pairwise distances between
  ///    these vertices are $\\le \\epsilon$.
  /// 3. The method iterates through all combinations of $k+1$ points for increasing $k$ (from $k=1$
  ///    up to number of points minus 1).
  /// 4. Distances are typically squared Euclidean distances for efficiency, so `epsilon` is squared
  ///    for comparison.
  pub fn build_complex(&self, cloud: &Cloud<N, F>, epsilon: F) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    let points_vec = cloud.points_ref();

    if points_vec.is_empty() {
      return complex;
    }

    // Add all points as 0-simplices (vertices).
    // The vertex indices in Simplex will correspond to indices in points_vec.
    for i in 0..points_vec.len() {
      complex.join_simplex(Simplex::new(0, vec![i]));
    }

    // Generate higher-dimensional simplices.
    // A k-simplex [p0, ..., pk] exists if all pairwise distances d(pi, pj) <= epsilon.
    // We iterate through simplex dimensions (dim_k), which is k.
    // A k-simplex has k+1 vertices.
    for dim_k in 1..points_vec.len() {
      // k from 1 up to num_points-1
      // Iterate over all combinations of (dim_k + 1) point indices
      for vertex_indices in (0..points_vec.len()).combinations(dim_k + 1) {
        let mut form_simplex = true;
        // Check pairwise distances for the current combination of points (potential simplex)
        // `vertex_indices` contains indices of points in `points_vec`.
        for edge_node_indices in vertex_indices.iter().copied().combinations(2) {
          let p1_idx = edge_node_indices[0];
          let p2_idx = edge_node_indices[1];

          // Retrieve the actual Vector points using the indices
          let point_a = points_vec[p1_idx];
          let point_b = points_vec[p2_idx];

          // Cloud::distance returns squared Euclidean distance.
          // Epsilon is the Euclidean distance threshold.
          // So, compare squared distance with epsilon * epsilon.
          if Cloud::<N, F>::distance(point_a, point_b) > epsilon * epsilon {
            form_simplex = false;
            break;
          }
        }

        if form_simplex {
          // If all pairs are within epsilon, this combination forms a simplex.
          // `vertex_indices` from `itertools::combinations` on a sorted range
          // will be sorted. `Simplex::new` also sorts, which is harmless.
          complex.join_simplex(Simplex::new(dim_k, vertex_indices));
        }
      }
    }
    complex
  }
}

/// Provides a default constructor for `VietorisRips` when the output is [`SimplicialComplex`].
impl<const N: usize, F> Default for VietorisRips<N, F, SimplicialComplex> {
  fn default() -> Self { Self::new() }
}

// TODO (autoparallel): The fact that we have an output param here is a bit weird. That API sucks
// and we should fix it.
/// Implements the [`Filtration`] trait for `VietorisRips` to generate a [`SimplicialComplex`].
///
/// This allows the `VietorisRips` struct to be used in contexts expecting a `Filtration`
/// that produces a simplicial complex for a given distance threshold `epsilon`.
impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> Filtration
  for VietorisRips<N, F, SimplicialComplex>
{
  type InputParameter = F;
  type InputSpace = Cloud<N, F>;
  type OutputParameter = ();
  type OutputSpace = SimplicialComplex;

  /// Builds the Vietoris-Rips [`SimplicialComplex`].
  ///
  /// This method delegates to [`VietorisRips::build_complex`].
  /// The `_output_param` is ignored for this implementation as the output
  /// is solely determined by the `epsilon` (`param`).
  fn build(
    &self,
    cloud: &Self::InputSpace,
    epsilon: Self::InputParameter,
    _output_param: &(), // Not used when output is SimplicialComplex
  ) -> Self::OutputSpace {
    self.build_complex(cloud, epsilon)
  }
}

/// Implements [`ParallelFiltration`] for `VietorisRips` targeting [`SimplicialComplex`] output.
///
/// This implementation is active when the `"parallel"` feature is enabled.
/// It signifies that the filtration construction process (which is primarily
/// [`VietorisRips::build_complex`]) could potentially be parallelized, although the current
/// `build_complex` itself might not be internally parallel. The trait marker is for higher-level
/// parallel processing of multiple filtration steps.
#[cfg(feature = "parallel")]
impl<const N: usize, F> ParallelFiltration for VietorisRips<N, F, SimplicialComplex>
where
  F: Field + Copy + Sum<F> + PartialOrd + Send + Sync,
  Cloud<N, F>: Sync,
  SimplicialComplex: Send,
{
  // No additional methods are required by ParallelFiltration if the base Filtration methods
  // are sufficient and types are Send + Sync.
  // build_parallel and build_serial will be available from the ParallelFiltration trait.
}

/// Implements the [`Filtration`] trait for `VietorisRips` to generate [`HomologyGroup`]s
/// for specified dimensions.
///
/// This allows the `VietorisRips` struct to be used to directly compute homology
/// for the Vietoris-Rips complex at a given `epsilon`.
///
/// # Type Parameters
/// * `R`: The coefficient [`Field`] for homology computations.
impl<const N: usize, F, R> Filtration for VietorisRips<N, F, HomologyGroup<R>>
where
  F: Field + Copy + Sum<F> + PartialOrd + Send + Sync, // Send + Sync for potential parallelism
  R: Field + Copy + Send + Sync,                       // Send + Sync for homology result
  Cloud<N, F>: Sync,
  SimplicialComplex: Send, // SimplicialComplex is built as an intermediate step
{
  type InputParameter = F;
  // Epsilon
  type InputSpace = Cloud<N, F>;
  type OutputParameter = HashSet<usize>;
  // Set of dimensions for which to compute homology
  type OutputSpace = HashMap<usize, HomologyGroup<R>>;

  // Map from dimension to HomologyGroup

  /// Builds the Vietoris-Rips complex for the given `input` (cloud) and `param` (epsilon),
  /// and then computes homology groups for the dimensions specified in `output_param`.
  ///
  /// # Arguments
  ///
  /// * `input`: The point [`Cloud<N, F>`].
  /// * `param`: The distance threshold `epsilon` of type `F`.
  /// * `output_param`: A `HashSet<usize>` specifying the dimensions for which homology groups
  ///   should be computed.
  ///
  /// # Returns
  ///
  /// A `HashMap<usize, HomologyGroup<R>>` where keys are dimensions and values are the
  /// computed [`HomologyGroup`]s with coefficients in `R`.
  fn build(
    &self,
    input: &Self::InputSpace,
    param: Self::InputParameter,          // epsilon
    output_param: &Self::OutputParameter, // dimensions for homology
  ) -> Self::OutputSpace {
    // First, build the Vietoris-Rips complex at the given epsilon.
    // This reuses the existing VietorisRips builder logic for SimplicialComplex.
    let complex_builder = VietorisRips::<N, F, SimplicialComplex>::new();
    let complex = complex_builder.build_complex(input, param);

    let mut homology_groups = HashMap::new();
    // For each dimension requested in output_param, compute homology.
    for dim in output_param {
      let homology_group = complex.compute_homology(*dim); // Pass R by type inference
      homology_groups.insert(*dim, homology_group);
    }
    homology_groups
  }
}

/// Implements [`ParallelFiltration`] for `VietorisRips` targeting [`HomologyGroup`] output.
///
/// This implementation is active when the `"parallel"` feature is enabled.
/// It signifies that the filtration construction and subsequent homology computations
/// can be processed in parallel, typically when dealing with a series of epsilon values.
/// The underlying `build` method for a single epsilon value involves building a complex
/// and then computing homology; these steps themselves might also have parallel potential
/// depending on their implementations.
#[cfg(feature = "parallel")]
impl<const N: usize, F, R> ParallelFiltration for VietorisRips<N, F, HomologyGroup<R>>
where
  F: Field + Copy + Sum<F> + PartialOrd + Send + Sync,
  R: Field + Copy + Send + Sync,
  Cloud<N, F>: Sync,
  HomologyGroup<R>: Send, /* Ensure the output homology groups can be sent across threads
                           * SimplicialComplex::Send is implicitly required by the Filtration
                           * impl above */
{
  // No additional methods are required by ParallelFiltration if the base Filtration methods
  // are sufficient and types are Send + Sync.
  // build_parallel and build_serial will be available from the ParallelFiltration trait.
}

#[cfg(test)]
mod tests {
  // For homology coefficients
  use harness_algebra::{
    algebras::boolean::Boolean, modular, prime_field, tensors::fixed::FixedVector,
  };

  use super::*;

  #[test]
  fn test_vietoris_rips_empty_cloud() {
    let cloud: Cloud<2, f64> = Cloud::new(vec![]);
    let vr = VietorisRips::<2, f64, SimplicialComplex>::new();
    let complex = vr.build(&cloud, 0.5, &());
    assert!(complex.simplices_by_dimension(0).is_none());
  }

  #[test]
  fn test_vietoris_rips_single_point() {
    let points = vec![FixedVector([0.0, 0.0])];
    let cloud = Cloud::new(points);
    let vr = VietorisRips::<2, f64, SimplicialComplex>::new();
    let complex = vr.build(&cloud, 0.5, &());

    let simplices_dim_0 = complex.simplices_by_dimension(0);
    assert_eq!(simplices_dim_0.unwrap().len(), 1);
    assert_eq!(simplices_dim_0.unwrap()[0].vertices(), &[0]);
    assert!(complex.simplices_by_dimension(1).is_none());
  }

  #[test]
  fn test_vietoris_rips_two_points() {
    let p1 = FixedVector([0.0, 0.0]);
    let p2 = FixedVector([1.0, 0.0]);
    let cloud = Cloud::new(vec![p1, p2]);
    let vr = VietorisRips::<2, f64, SimplicialComplex>::new();

    // Epsilon too small for an edge
    let complex_no_edge = vr.build(&cloud, 0.5, &()); // distance is 1.0
    assert_eq!(complex_no_edge.simplices_by_dimension(0).unwrap().len(), 2);
    assert!(complex_no_edge.simplices_by_dimension(1).is_none());

    // Epsilon large enough for an edge
    let complex_with_edge = vr.build(&cloud, 1.5, &());
    assert_eq!(complex_with_edge.simplices_by_dimension(0).unwrap().len(), 2);
    let simplices_dim_1 = complex_with_edge.simplices_by_dimension(1);
    assert_eq!(simplices_dim_1.unwrap().len(), 1);
    assert_eq!(simplices_dim_1.unwrap()[0].vertices(), &[0, 1]);
  }

  #[test]
  fn test_vietoris_rips_triangle() {
    let p0 = FixedVector([0.0, 0.0]);
    let p1 = FixedVector([1.0, 0.0]);
    let p2 = FixedVector([0.5, 0.866]); // Equilateral triangle, side length 1

    let cloud = Cloud::new(vec![p0, p1, p2]);
    let vr = VietorisRips::<2, f64, SimplicialComplex>::new();

    // Distances: d(p0,p1)=1, d(p0,p2) approx 1, d(p1,p2) approx 1
    // Norm of p0-p1: (-1)^2 + 0^2 = 1
    // Norm of p0-p2: (-0.5)^2 + (-0.866)^2 = 0.25 + 0.749956 = 0.999956 approx 1
    // Norm of p1-p2: (0.5)^2 + (-0.866)^2 = 0.25 + 0.749956 = 0.999956 approx 1

    // Epsilon just enough for edges, not for triangle (if strict definition used sometimes, but
    // here diameter based) Here, if all edges are present, the triangle [0,1,2] will be added
    // by `join_simplex` for the 2-simplex.
    let complex = vr.build(&cloud, 1.1, &());

    assert_eq!(complex.simplices_by_dimension(0).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(1).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(2).unwrap().len(), 1);
    assert_eq!(complex.simplices_by_dimension(2).unwrap()[0].vertices(), &[0, 1, 2]);
  }

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);

  #[test]
  fn test_compute_homology_filtration_basic() {
    let p0 = FixedVector([0.0, 0.0]);
    let p1 = FixedVector([1.0, 0.0]);
    let cloud: Cloud<2, f64> = Cloud::new(vec![p0, p1]);
    let vr_builder = VietorisRips::<2, f64, HomologyGroup<Mod7>>::new();

    let epsilons = vec![0.5, 1.5]; // Epsilon_0: 2 components, Epsilon_1: 1 component
    let dims = HashSet::from([0, 1]);

    // Assuming ParallelFiltration trait provides build_serial if not parallel feature
    // Or, if not, we'd call vr_builder.build(&cloud, epsilon, &dims) for each epsilon
    // For now, let's assume build_serial is available or a similar helper exists.
    // If this test should run without "parallel", we need to adjust how homology_results are
    // obtained. The original code used `build_serial`. Let's make it work for both cases
    // by calling `build` in a loop.

    let mut homology_results_vec = Vec::new();
    for &epsilon in &epsilons {
      homology_results_vec.push(vr_builder.build(&cloud, epsilon, &dims));
    }
    let homology_results = &homology_results_vec;

    assert_eq!(homology_results.len(), 2);

    // Check results for epsilon = 0.5
    let homology_at_eps0_5 = &homology_results[0];
    let h0_eps0_5 = homology_at_eps0_5.get(&0).unwrap();
    assert_eq!(h0_eps0_5.dimension, 0);
    assert_eq!(h0_eps0_5.betti_number, 2, "H0(eps=0.5): Expected 2 connected components");

    let h1_eps0_5 = homology_at_eps0_5.get(&1).unwrap();
    assert_eq!(h1_eps0_5.dimension, 1);
    assert_eq!(h1_eps0_5.betti_number, 0, "H1(eps=0.5): Expected 0 1-cycles");

    // Check results for epsilon = 1.5 (two points, one edge -> one component)
    let homology_at_eps1_5 = &homology_results[1];
    let h0_eps1_5 = homology_at_eps1_5.get(&0).unwrap();
    assert_eq!(h0_eps1_5.dimension, 0);
    assert_eq!(h0_eps1_5.betti_number, 1, "H0(eps=1.5): Expected 1 connected component");

    let h1_eps1_5 = homology_at_eps1_5.get(&1).unwrap();
    assert_eq!(h1_eps1_5.dimension, 1);
    assert_eq!(
      h1_eps1_5.betti_number, 0,
      "H1(eps=1.5): Expected 0 1-cycles (edge is contractible)"
    );
  }

  #[cfg(feature = "parallel")]
  #[test]
  fn test_compute_homology_filtration_parallel_triangle() {
    // This test runs only if 'parallel' feature is enabled.
    // It implicitly uses build_parallel from the ParallelFiltration trait.
    use crate::filtration::ParallelFiltration; // Make sure trait is in scope

    let p0 = FixedVector([0.0, 0.0]);
    let p1 = FixedVector([1.0, 0.0]);
    let p2 = FixedVector([0.5, 0.8660254]); // Equilateral triangle, side length 1.0

    let cloud = Cloud::new(vec![p0, p1, p2]);
    let vr_builder = VietorisRips::<2, f64, HomologyGroup<Boolean>>::new();
    // Distances: d(p0,p1)=1, d(p0,p2)=1, d(p1,p2)=1
    let epsilons = vec![0.5, 1.1];
    // eps=0.5: 3 points (3 components in H0)
    // eps=1.1: All edges [0,1],[1,2],[0,2] exist (all distances are 1.0 <= 1.1).
    //          VR complex includes the 2-simplex [0,1,2] because all pairwise distances are <=
    //          1.1.
    //         So, it's a filled triangle. H0=1, H1=0, H2=0.
    let dims = HashSet::from([0, 1, 2]);

    // This uses build_parallel from ParallelFiltration trait
    let homology_results = vr_builder.build_parallel(&cloud, epsilons, &dims);

    assert_eq!(homology_results.len(), 2);

    // Results for epsilon = 0.5
    let homology_at_eps0_5 = &homology_results[0];
    let h0_eps0_5 = homology_at_eps0_5.get(&0).unwrap();
    assert_eq!(h0_eps0_5.dimension, 0);
    assert_eq!(h0_eps0_5.betti_number, 3, "H0(eps=0.5) for triangle points");
    let h1_eps0_5 = homology_at_eps0_5.get(&1).unwrap();
    assert_eq!(h1_eps0_5.betti_number, 0, "H1(eps=0.5) for triangle points");
    let h2_eps0_5 = homology_at_eps0_5.get(&2).unwrap();
    assert_eq!(h2_eps0_5.betti_number, 0, "H2(eps=0.5) for triangle points");

    // Results for epsilon = 1.1 (filled triangle)
    let homology_at_eps1_1 = &homology_results[1];
    let h0_eps1_1 = homology_at_eps1_1.get(&0).unwrap();
    assert_eq!(h0_eps1_1.dimension, 0);
    assert_eq!(h0_eps1_1.betti_number, 1, "H0(eps=1.1) for filled triangle");
    let h1_eps1_1 = homology_at_eps1_1.get(&1).unwrap();
    assert_eq!(h1_eps1_1.betti_number, 0, "H1(eps=1.1) for filled triangle");
    let h2_eps1_1 = homology_at_eps1_1.get(&2).unwrap();
    assert_eq!(h2_eps1_1.betti_number, 0, "H2(eps=1.1) for filled triangle");
  }
}
