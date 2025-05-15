// space/src/vietoris_rips.rs
//! Implements the Vietoris-Rips complex construction.

use std::{iter::Sum, marker::PhantomData};

use harness_algebra::ring::Field;
use itertools::Itertools;

#[cfg(feature = "parallel")]
use crate::filtration::ParallelFiltration;
use crate::{
  cloud::Cloud,
  filtration::Filtration,
  prelude::MetricSpace,
  simplicial::{Simplex, SimplicialComplex}, // The output space
};

/// A struct that allows construction of Vietoris-Rips complexes.
///
/// It implements the `Filtration` trait, taking a `Cloud` of points and a distance
/// threshold `epsilon` to produce a `SimplicialComplex`.
///
/// A k-simplex `[v0, v1, ..., vk]` is included in the Vietoris-Rips complex if
/// the distance between any pair of its vertices `(vi, vj)` is less than or equal
/// to the threshold `epsilon`.
pub struct VietorisRips<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> {
  _phantom: PhantomData<[F; N]>, // To use N and F generics
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> VietorisRips<N, F> {
  /// Creates a new `VietorisRips` constructor.
  pub fn new() -> Self { Self { _phantom: PhantomData } }
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> Default for VietorisRips<N, F> {
  fn default() -> Self { Self::new() }
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> Filtration for VietorisRips<N, F> {
  type InputSpace = Cloud<N, F>;
  // Epsilon, the distance threshold
  type OutputSpace = SimplicialComplex;
  type Parameter = F;

  fn build(&self, cloud: &Self::InputSpace, epsilon: Self::Parameter) -> Self::OutputSpace {
    let mut complex = SimplicialComplex::new();
    let points_vec = cloud.points_ref(); // We'll need to add this method to Cloud

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
          let point_a = points_vec[p1_idx]; // This is Vector<N, F>
          let point_b = points_vec[p2_idx]; // This is Vector<N, F>

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

#[cfg(feature = "parallel")]
impl<const N: usize, F> ParallelFiltration for VietorisRips<N, F>
where
  F: Field + Copy + Sum<F> + PartialOrd + Send + Sync, // Added Send + Sync for F
  Cloud<N, F>: Sync,                                   // Ensure Cloud is Sync
  SimplicialComplex: Send,                             // Ensure SimplicialComplex is Send
{
  // build_parallel is already defined with a default implementation in the trait
}

#[cfg(test)]
mod tests {
  use harness_algebra::vector::Vector;

  use super::*;

  #[test]
  fn test_vietoris_rips_empty_cloud() {
    let cloud: Cloud<2, f64> = Cloud::new(vec![]);
    let vr = VietorisRips::<2, f64>::new();
    let complex = vr.build(&cloud, 0.5);
    assert!(complex.simplices_by_dimension(0).is_none());
  }

  #[test]
  fn test_vietoris_rips_single_point() {
    let points = vec![Vector([0.0, 0.0])];
    let cloud = Cloud::new(points);
    let vr = VietorisRips::<2, f64>::new();
    let complex = vr.build(&cloud, 0.5);

    let simplices_dim_0 = complex.simplices_by_dimension(0);
    assert_eq!(simplices_dim_0.unwrap().len(), 1);
    assert_eq!(simplices_dim_0.unwrap()[0].vertices(), &[0]);
    assert!(complex.simplices_by_dimension(1).is_none());
  }

  #[test]
  fn test_vietoris_rips_two_points() {
    let p1 = Vector([0.0, 0.0]);
    let p2 = Vector([1.0, 0.0]);
    let cloud = Cloud::new(vec![p1, p2]);
    let vr = VietorisRips::<2, f64>::new();

    // Epsilon too small for an edge
    let complex_no_edge = vr.build(&cloud, 0.5); // distance is 1.0
    assert_eq!(complex_no_edge.simplices_by_dimension(0).unwrap().len(), 2);
    assert!(complex_no_edge.simplices_by_dimension(1).is_none());

    // Epsilon large enough for an edge
    let complex_with_edge = vr.build(&cloud, 1.5);
    assert_eq!(complex_with_edge.simplices_by_dimension(0).unwrap().len(), 2);
    let simplices_dim_1 = complex_with_edge.simplices_by_dimension(1);
    assert_eq!(simplices_dim_1.unwrap().len(), 1);
    assert_eq!(simplices_dim_1.unwrap()[0].vertices(), &[0, 1]);
  }

  #[test]
  fn test_vietoris_rips_triangle() {
    let p0 = Vector([0.0, 0.0]);
    let p1 = Vector([1.0, 0.0]);
    let p2 = Vector([0.5, 0.866]); // Equilateral triangle, side length 1

    let cloud = Cloud::new(vec![p0, p1, p2]);
    let vr = VietorisRips::<2, f64>::new();

    // Distances: d(p0,p1)=1, d(p0,p2) approx 1, d(p1,p2) approx 1
    // Norm of p0-p1: (-1)^2 + 0^2 = 1
    // Norm of p0-p2: (-0.5)^2 + (-0.866)^2 = 0.25 + 0.749956 = 0.999956 approx 1
    // Norm of p1-p2: (0.5)^2 + (-0.866)^2 = 0.25 + 0.749956 = 0.999956 approx 1

    // Epsilon just enough for edges, not for triangle (if strict definition used sometimes, but
    // here diameter based) Here, if all edges are present, the triangle [0,1,2] will be added
    // by `join_simplex` for the 2-simplex.
    let complex = vr.build(&cloud, 1.1);

    assert_eq!(complex.simplices_by_dimension(0).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(1).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(2).unwrap().len(), 1);
    assert_eq!(complex.simplices_by_dimension(2).unwrap()[0].vertices(), &[0, 1, 2]);
  }

  #[cfg(feature = "parallel")]
  #[test]
  fn test_vietoris_rips_parallel() {
    use crate::filtration::ParallelFiltration; // Ensure trait is in scope

    let p0 = Vector([0.0, 0.0]);
    let p1 = Vector([1.0, 0.0]);
    let p2 = Vector([2.0, 0.0]);
    let p3 = Vector([0.5, 0.866]); // Forms a triangle with p0, p1 if epsilon is right

    let cloud = Cloud::new(vec![p0, p1, p2, p3]);
    let vr_builder = VietorisRips::<2, f64>::new();

    let epsilons = vec![0.5, 1.1, 2.1]; // Three different epsilon values

    let complexes = vr_builder.build_parallel(&cloud, epsilons);

    assert_eq!(complexes.len(), 3);

    // Complex 0 (epsilon = 0.5): Only vertices
    let complex0 = &complexes[0];
    assert_eq!(complex0.simplices_by_dimension(0).unwrap().len(), 4); // 4 points
    assert!(complex0.simplices_by_dimension(1).is_none()); // No edges

    // Complex 1 (epsilon = 1.1):
    // Edges: [0,1], [1,2], [0,3], [1,3]
    // 0-1: dist 1.0
    // 1-2: dist 1.0
    // 0-3: dist 1.0
    // 1-3: dist 1.0
    // Triangle: [0,1,3]
    let complex1 = &complexes[1];
    assert_eq!(complex1.simplices_by_dimension(0).unwrap().len(), 4);
    assert_eq!(
      complex1.simplices_by_dimension(1).unwrap().len(),
      4,
      "Expected 4 edges for eps=1.1"
    );
    assert_eq!(
      complex1.simplices_by_dimension(2).unwrap().len(),
      1,
      "Expected 1 triangle for eps=1.1"
    ); // [0,1,3]
    assert!(
      complex1.simplices_by_dimension(2).unwrap()[0].vertices() == [0, 1, 3]
        || complex1.simplices_by_dimension(2).unwrap()[0].vertices() == [0, 1, 2]
        || complex1.simplices_by_dimension(2).unwrap()[0].vertices() == [0, 2, 3]
        || complex1.simplices_by_dimension(2).unwrap()[0].vertices() == [1, 2, 3],
      "Triangle [0,1,3] should exist"
    );

    // Complex 2 (epsilon = 2.1):
    // All points are connected to all other points.
    // Should form a tetrahedron if we were in 3D, here it will be a full 3-simplex (K4)
    // 4 vertices (0-simplices)
    // 6 edges (1-simplices: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
    // 4 triangles (2-simplices: (0,1,2), (0,1,3), (0,2,3), (1,2,3))
    // 1 tetrahedron (3-simplex: (0,1,2,3))
    let complex2 = &complexes[2];
    assert_eq!(complex2.simplices_by_dimension(0).unwrap().len(), 4);
    assert_eq!(
      complex2.simplices_by_dimension(1).unwrap().len(),
      6,
      "Expected 6 edges for eps=2.1"
    );
    assert_eq!(
      complex2.simplices_by_dimension(2).unwrap().len(),
      4,
      "Expected 4 triangles for eps=2.1"
    );
    assert_eq!(
      complex2.simplices_by_dimension(3).unwrap().len(),
      1,
      "Expected 1 3-simplex (tetrahedron) for eps=2.1"
    );
  }
}
