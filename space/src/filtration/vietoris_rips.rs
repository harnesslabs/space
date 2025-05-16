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
  simplicial::{HomologyGroup, Simplex, SimplicialComplex}, // Added HomologyGroup
};

pub trait VROutputSpace: Send + Sync {}
impl VROutputSpace for SimplicialComplex {}
impl<F: Field + Copy + Sum<F> + PartialOrd + Send + Sync> VROutputSpace for HomologyGroup<F> {}

/// A struct that allows construction of Vietoris-Rips complexes.
///
/// It implements the `Filtration` trait, taking a `Cloud` of points and a distance
/// threshold `epsilon` to produce a `SimplicialComplex`.
///
/// A k-simplex `[v0, v1, ..., vk]` is included in the Vietoris-Rips complex if
/// the distance between any pair of its vertices `(vi, vj)` is less than or equal
/// to the threshold `epsilon`.
pub struct VietorisRips<const N: usize, F: Field + Copy + Sum<F> + PartialOrd, O: VROutputSpace> {
  _phantom:      PhantomData<[F; N]>, // To use N and F generics
  _output_space: PhantomData<O>,
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd, O: VROutputSpace>
  VietorisRips<N, F, O>
{
  /// Creates a new `VietorisRips` constructor.
  pub fn new() -> Self { Self { _phantom: PhantomData, _output_space: PhantomData } }

  pub fn build_complex(&self, cloud: &Cloud<N, F>, epsilon: F) -> SimplicialComplex {
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

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd> Default
  for VietorisRips<N, F, SimplicialComplex>
{
  fn default() -> Self { Self::new() }
}

impl<const N: usize, F: Field + Copy + Sum<F> + PartialOrd, O: VROutputSpace> Filtration
  for VietorisRips<N, F, O>
{
  type InputSpace = Cloud<N, F>;
  // Epsilon, the distance threshold
  type OutputSpace = O;
  type Parameter = F;

  fn build(&self, cloud: &Self::InputSpace, epsilon: Self::Parameter) -> Self::OutputSpace {
    let complex = self.build_complex(cloud, epsilon);
    match self._output_space {
      PhantomData::<SimplicialComplex> => complex,
      _ => todo!(),
    }
  }
}

#[cfg(feature = "parallel")]
impl<const N: usize, F> ParallelFiltration for VietorisRips<N, F>
where
  F: Field + Copy + Sum<F> + PartialOrd + Send + Sync,
  Cloud<N, F>: Sync,
  SimplicialComplex: Send,
{
}

#[cfg(test)]
mod tests {
  use harness_algebra::arithmetic::Boolean; // For homology coefficients
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

  #[test]
  fn test_compute_homology_filtration_basic() {
    let p0 = Vector([0.0, 0.0]);
    let p1 = Vector([1.0, 0.0]);
    let cloud: Cloud<2, f64> = Cloud::new(vec![p0, p1]);
    let vr_builder = VietorisRips::<2, f64>::new();

    let epsilons = vec![0.5, 1.5]; // Epsilon_0: 2 components, Epsilon_1: 1 component
    let max_dim = 1;

    let homology_results =
      vr_builder.compute_homology_filtration::<Boolean>(&cloud, epsilons.clone(), max_dim);

    assert_eq!(homology_results.len(), 2);

    // Check results for epsilon = 0.5
    let (eps0, hg_eps0) = &homology_results[0];
    assert_eq!(*eps0, 0.5);
    assert_eq!(hg_eps0.len(), max_dim + 1, "Expected H0 and H1");

    // H0 for epsilon = 0.5 (two points, no edge)
    let h0_eps0 = &hg_eps0[0];
    assert_eq!(h0_eps0.dimension, 0);
    assert_eq!(h0_eps0.betti_number, 2, "H0(eps=0.5): Expected 2 connected components");

    // H1 for epsilon = 0.5
    let h1_eps0 = &hg_eps0[1];
    assert_eq!(h1_eps0.dimension, 1);
    assert_eq!(h1_eps0.betti_number, 0, "H1(eps=0.5): Expected 0 1-cycles");

    // Check results for epsilon = 1.5 (two points, one edge -> one component)
    let (eps1, hg_eps1) = &homology_results[1];
    assert_eq!(*eps1, 1.5);
    assert_eq!(hg_eps1.len(), max_dim + 1);

    // H0 for epsilon = 1.5
    let h0_eps1 = &hg_eps1[0];
    assert_eq!(h0_eps1.dimension, 0);
    assert_eq!(h0_eps1.betti_number, 1, "H0(eps=1.5): Expected 1 connected component");

    // H1 for epsilon = 1.5
    let h1_eps1 = &hg_eps1[1];
    assert_eq!(h1_eps1.dimension, 1);
    assert_eq!(h1_eps1.betti_number, 0, "H1(eps=1.5): Expected 0 1-cycles (edge is contractible)");
  }

  #[cfg(feature = "parallel")]
  #[test]
  fn test_compute_homology_filtration_parallel_triangle() {
    // This test runs only if 'parallel' feature is enabled.
    // It implicitly uses build_parallel inside compute_homology_filtration.
    // use crate::filtration::ParallelFiltration; // Not strictly needed here if only using the new
    // method

    let p0 = Vector([0.0, 0.0]);
    let p1 = Vector([1.0, 0.0]);
    let p2 = Vector([0.5, 0.8660254]); // Equilateral triangle, side length 1.0

    let cloud = Cloud::new(vec![p0, p1, p2]);
    let vr_builder = VietorisRips::<2, f64>::new();
    // Distances: d(p0,p1)=1, d(p0,p2)=1, d(p1,p2)=1
    let epsilons = vec![0.5, 1.1];
    // eps=0.5: 3 points (3 components in H0)
    // eps=1.1: All edges [0,1],[1,2],[0,2] exist (all distances are 1.0 <= 1.1).
    //          VR complex includes the 2-simplex [0,1,2] because all pairwise distances are <= 1.1.
    //          So, it's a filled triangle. H0=1, H1=0, H2=0.
    let max_dim = 2;

    let homology_results =
      vr_builder.compute_homology_filtration::<Boolean>(&cloud, epsilons.clone(), max_dim);

    assert_eq!(homology_results.len(), 2);

    // Results for epsilon = 0.5
    let (eps0_val, hg_eps0) = &homology_results[0];
    assert_eq!(*eps0_val, 0.5);
    assert_eq!(hg_eps0.len(), max_dim + 1);
    assert_eq!(hg_eps0[0].betti_number, 3, "H0(eps=0.5) for triangle points"); // H0: 3 components
    assert_eq!(hg_eps0[1].betti_number, 0, "H1(eps=0.5) for triangle points"); // H1: 0 holes
    assert_eq!(hg_eps0[2].betti_number, 0, "H2(eps=0.5) for triangle points"); // H2: 0 voids

    // Results for epsilon = 1.1 (filled triangle)
    let (eps1_val, hg_eps1) = &homology_results[1];
    assert_eq!(*eps1_val, 1.1);
    assert_eq!(hg_eps1.len(), max_dim + 1);
    assert_eq!(hg_eps1[0].betti_number, 1, "H0(eps=1.1) for filled triangle"); // H0: 1 component
    assert_eq!(hg_eps1[1].betti_number, 0, "H1(eps=1.1) for filled triangle"); // H1: 0 holes
    assert_eq!(hg_eps1[2].betti_number, 0, "H2(eps=1.1) for filled triangle"); // H2: 0 voids (it's
                                                                               // a 2-simplex, not
                                                                               // hollow)
  }
}
