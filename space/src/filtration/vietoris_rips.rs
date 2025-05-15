// space/src/vietoris_rips.rs
//! Implements the Vietoris-Rips complex construction.

use std::{iter::Sum, marker::PhantomData};

use harness_algebra::{ring::Field, vector::Vector}; /* Assuming Vector is needed for
                                                      * Cloud's points */
use itertools::Itertools; // For combinations

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

          // Use the static distance method from Cloud's MetricSpace impl
          if Cloud::<N, F>::distance(point_a, point_b) > epsilon {
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

#[cfg(test)]
mod tests {
  use harness_algebra::vector::Vector;

  use super::*;
  use crate::definitions::MetricSpace; // for Cloud::distance

  #[test]
  fn test_vietoris_rips_empty_cloud() {
    let cloud: Cloud<2, f64> = Cloud::new(vec![]);
    let vr = VietorisRips::<2, f64>::new();
    let complex = vr.build(&cloud, 0.5);
    assert!(complex.simplices_by_dimension(0).is_empty());
  }

  #[test]
  fn test_vietoris_rips_single_point() {
    let points = vec![Vector([0.0, 0.0])];
    let cloud = Cloud::new(points);
    let vr = VietorisRips::<2, f64>::new();
    let complex = vr.build(&cloud, 0.5);

    let simplices_dim_0 = complex.simplices_by_dimension(0);
    assert_eq!(simplices_dim_0.len(), 1);
    assert_eq!(simplices_dim_0[0].vertices(), &[0]);
    assert!(complex.simplices_by_dimension(1).is_empty());
  }

  #[test]
  fn test_vietoris_rips_two_points() {
    let p1 = Vector([0.0, 0.0]);
    let p2 = Vector([1.0, 0.0]);
    let cloud = Cloud::new(vec![p1, p2]);
    let vr = VietorisRips::<2, f64>::new();

    // Epsilon too small for an edge
    let complex_no_edge = vr.build(&cloud, 0.5); // distance is 1.0
    assert_eq!(complex_no_edge.simplices_by_dimension(0).len(), 2);
    assert!(complex_no_edge.simplices_by_dimension(1).is_empty());

    // Epsilon large enough for an edge
    let complex_with_edge = vr.build(&cloud, 1.5);
    assert_eq!(complex_with_edge.simplices_by_dimension(0).len(), 2);
    let simplices_dim_1 = complex_with_edge.simplices_by_dimension(1);
    assert_eq!(simplices_dim_1.len(), 1);
    assert_eq!(simplices_dim_1[0].vertices(), &[0, 1]);
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

    assert_eq!(complex.simplices_by_dimension(0).len(), 3);
    assert_eq!(complex.simplices_by_dimension(1).len(), 3);
    assert_eq!(complex.simplices_by_dimension(2).len(), 1);
    assert_eq!(complex.simplices_by_dimension(2)[0].vertices(), &[0, 1, 2]);
  }
  // To run these tests, SimplicialComplex needs a way to access simplices by dimension,
  // e.g., a method like `simplices_by_dimension(&self, dim: usize) -> Option<&Vec<Simplex>>`.
  // Add this to SimplicialComplex:
  // pub fn simplices_by_dimension(&self, dim: usize) -> Option<&Vec<Simplex>> {
  //   self.simplices.get(dim)
  // }
}
