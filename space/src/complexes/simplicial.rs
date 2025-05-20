//! # Simplicial Topology: Complexes, Chains, and Homology
//!
//! This module provides a suite of tools for working with simplicial complexes, a fundamental
//! concept in algebraic topology. Simplicial complexes are used to represent topological spaces by
//! breaking them down into simple building blocks called simplices (points, line segments,
//! triangles, tetrahedra, and their higher-dimensional counterparts).
//!
//! ## Core Concepts
//!
//! - **Simplices**: Represented by the [`Simplex`] struct. A $k$-simplex is the convex hull of
//!   $k+1$ affinely independent points. Vertices in our implementation are always stored as sorted
//!   `usize` indices.
//!   - $0$-simplex: a point (e.g., $v_0$)
//!   - $1$-simplex: a line segment (e.g., $(v_0, v_1)$)
//!   - $2$-simplex: a triangle (e.g., $(v_0, v_1, v_2)$)
//!   - $3$-simplex: a tetrahedron (e.g., $(v_0, v_1, v_2, v_3)$)
//!
//! - **Simplicial Complex**: Represented by the [`SimplicialComplex`] struct. This is a collection
//!   of simplices that is closed under taking faces (i.e., if a simplex is in the complex, all its
//!   faces must also be in the complex) and such that the intersection of any two simplices is
//!   either empty or a face of both.
//!
//! - **Chains**: Represented by the [`Chain<R>`] struct. A $k$-chain is a formal sum of
//!   $k$-simplices with coefficients in a ring $R$. For example, $c = 3\sigma_1 - 2\sigma_2 +
//!   \sigma_3$, where $\sigma_i$ are $k$-simplices.
//!
//! - **Boundary Operator**: The boundary operator $\partial_k: C_k(X; R) \to C_{k-1}(X; R)$ maps a
//!   $k$-chain to a $(k-1)$-chain. For a single $k$-simplex $\sigma = [v_0, v_1, \dots, v_k]$, its
//!   boundary is defined as: $ \partial_k \sigma = \sum_{i=0}^{k} (-1)^i [v_0, \dots, \hat{v_i},
//!   \dots, v_k] $ where $\hat{v_i}$ means the vertex $v_i$ is omitted. A crucial property is that
//!   $\partial_{k-1} \circ \partial_k = 0$ (the boundary of a boundary is zero).
//!
//! - **Homology Groups**: Represented by the [`HomologyGroup<F>`] struct. The $k$-th homology group
//!   $H_k(X; F)$ with coefficients in a field $F$ is defined as the quotient group: $ H_k(X; F) =
//!   Z_k(X; F) / B_k(X; F) $ where:
//!     - $Z_k(X; F) = \ker \partial_k$ is the group of $k$-cycles (chains whose boundary is zero).
//!     - $B_k(X; F) = \text{im } \partial_{k+1}$ is the group of $k$-boundaries (chains that are
//!       boundaries of $(k+1)$-chains).
//!   - Homology groups are important topological invariants that capture information about the
//!     \"holes\" of different dimensions in a space. The rank of $H_k(X; F)$ is called the $k$-th
//!     Betti number, $b_k$.
//!
//! ## Features
//!
//! - Construction and manipulation of [`Simplex`] and [`SimplicialComplex`] objects.
//! - Computation of simplex faces.
//! - Implementation of [`Chain`] arithmetic (addition).
//! - Standard boundary operator $\partial$ for [`Chain`]s.
//! - Calculation of homology groups [`HomologyGroup<F>`] using Gaussian elimination over a generic
//!   [`Field`] `F`. This involves:
//!     - Constructing boundary matrices.
//!     - Computing kernel and image bases of these matrices using
//!       [`DynamicDenseMatrix::row_echelon_form`].
//!
//! ## Usage Example
//!
//! ```rust
//! use harness_algebra::algebras::boolean::Boolean;
//! use harness_space::complexes::simplicial::{Simplex, SimplicialComplex};
//!
//! // Create a simplicial complex representing a hollow triangle (cycle C3)
//! let mut complex = SimplicialComplex::new();
//! complex.join_simplex(Simplex::new(1, vec![0, 1])); // Edge (0,1)
//! complex.join_simplex(Simplex::new(1, vec![1, 2])); // Edge (1,2)
//! complex.join_simplex(Simplex::new(1, vec![2, 0])); // Edge (2,0)
//!
//! // Compute H_0 and H_1 with Z/2Z coefficients
//! let h0 = complex.homology::<Boolean>(0);
//! let h1 = complex.homology::<Boolean>(1);
//!
//! assert_eq!(h0.betti_number, 1); // One connected component
//! assert_eq!(h1.betti_number, 1); // One 1-dimensional hole (the triangle itself)
//! ```
//!
//! ## Further Reading
//!
//! For more information on simplicial complexes and homology theory, consider these resources:
//! - Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press. (Especially Chapter 2)
//! - Munkres, J. R. (1984). *Elements of Algebraic Topology*. Addison-Wesley.

use std::collections::HashMap;

use harness_algebra::tensors::dynamic::{
  compute_quotient_basis,
  matrix::{DynamicDenseMatrix, RowMajor},
};
use itertools::Itertools;

use super::*;
use crate::{
  definitions::Topology,
  homology::{Chain, Homology},
};

/// A simplex represents a $k$-dimensional geometric object, defined as the convex hull of $k+1$
/// affinely independent vertices.
///
/// Simplices are the basic building blocks of [`SimplicialComplex`]es.
/// - A 0-simplex (dimension 0) is a point.
/// - A 1-simplex (dimension 1) is a line segment.
/// - A 2-simplex (dimension 2) is a triangle.
/// - A 3-simplex (dimension 3) is a tetrahedron.
/// - ... and so on for higher dimensions.
///
/// In this implementation, vertices are represented by `usize` indices and are always stored in a
/// sorted vector to ensure a canonical representation for each simplex.
///
/// # Fields
/// * `vertices`: A sorted `Vec<usize>` of vertex indices that define the simplex.
/// * `dimension`: The dimension of the simplex, equal to `vertices.len() - 1`.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Simplex {
  vertices:  Vec<usize>,
  dimension: usize,
}

impl Eq for Simplex {}

impl PartialOrd for Simplex {
  /// Provides a partial ordering for simplices.
  ///
  /// This implementation delegates to `cmp`, thus providing a total ordering.
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Simplex {
  /// Provides a total ordering for simplices, primarily for use in sorted collections (e.g.,
  /// `BTreeSet` or when sorting `Vec<Simplex>`).
  ///
  /// The ordering is based on the lexicographical comparison of their sorted vertex lists.
  fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.vertices.cmp(&other.vertices) }
}

impl Simplex {
  /// Creates a new simplex of a given `dimension` from a set of `vertices`.
  ///
  /// The provided `vertices` will be sorted internally to ensure a canonical representation.
  ///
  /// # Arguments
  /// * `dimension`: The dimension of the simplex (e.g., 0 for a point, 1 for an edge, 2 for a
  ///   triangle).
  /// * `vertices`: A vector of `usize` vertex indices. The length of this vector must be `dimension
  ///   + 1`.
  ///
  /// # Panics
  /// * If `vertices.len()` does not equal `dimension + 1`.
  /// * If any vertex indices in `vertices` are repeated (i.e., vertices are not distinct).
  pub fn new(dimension: usize, vertices: Vec<usize>) -> Self {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices: vertices.into_iter().sorted().collect(), dimension }
  }

  /// Returns a slice reference to the sorted vertex indices of the simplex.
  pub fn vertices(&self) -> &[usize] { &self.vertices }

  /// Returns the dimension of the simplex.
  ///
  /// The dimension $k$ is equal to the number of vertices minus one.
  pub const fn dimension(&self) -> usize { self.dimension }

  /// Computes all $(k-1)$-dimensional faces of this $k$-simplex.
  ///
  /// A face is obtained by removing one vertex from the simplex's vertex set.
  /// For example:
  /// - The faces of a 2-simplex (triangle) `[v_0, v_1, v_2]` are its three 1-simplices (edges):
  ///   `[v_1, v_2]`, `[v_0, v_2]`, and `[v_0, v_1]`.
  /// - The faces of a 1-simplex (edge) `[v_0, v_1]` are its two 0-simplices (vertices): `[v_1]` and
  ///   `[v_0]`.
  /// - A 0-simplex has no faces (its dimension is -1, typically considered an empty set of faces).
  ///
  /// # Returns
  /// A [`Vec<Simplex>`] containing all $(k-1)$-dimensional faces. If the simplex is 0-dimensional,
  /// an empty vector is returned as it has no $( -1)$-dimensional faces in the typical sense.
  pub fn faces(&self) -> Vec<Self> {
    self
      .vertices
      .clone()
      .into_iter()
      .combinations(self.dimension)
      .map(|v| Self::new(self.dimension - 1, v))
      .collect()
  }
}

/// A simplicial complex $K$ is a collection of simplices satisfying two conditions:
/// 1. Every face of a simplex in $K$ is also in $K$.
/// 2. The intersection of any two simplices in $K$ is either empty or a face of both.
///
/// This struct stores simplices grouped by their dimension in a `HashMap`.
/// The `join_simplex` method ensures that when a simplex is added, all its faces are also added
/// recursively, maintaining the first condition.
#[derive(Debug, Default)]
pub struct SimplicialComplex {
  /// A map from dimension to a vector of simplices of that dimension.
  /// Simplices within each dimension are not guaranteed to be sorted after arbitrary joins,
  /// but `compute_homology` sorts them internally as needed.
  simplices: HashMap<usize, Vec<Simplex>>,
}

impl SimplicialComplex {
  /// Creates a new, empty simplicial complex.
  pub fn new() -> Self { Self { simplices: HashMap::new() } }

  /// Adds a simplex to the complex. If the simplex is already present, it is not added again.
  ///
  /// Crucially, this method also recursively adds all faces of the given `simplex` to the complex,
  /// ensuring that the definition of a simplicial complex (closure under faces) is maintained.
  ///
  /// # Arguments
  /// * `simplex`: The [`Simplex`] to add to the complex.
  pub fn join_simplex(&mut self, simplex: Simplex) {
    let dim = simplex.dimension();
    let simplices_in_dim = self.simplices.entry(dim).or_default();

    if simplices_in_dim.contains(&simplex) {
      return;
    }

    if simplex.dimension() > 0 {
      for face in simplex.faces() {
        self.join_simplex(face); // Recursive call
      }
    }
    // Add the current simplex after its faces (if any) are processed.
    // This re-fetches mutable access in case recursion modified other dimensions.
    self.simplices.entry(dim).or_default().push(simplex);
  }

  /// Returns a slice reference to the simplices of a given `dimension` stored in the complex.
  ///
  /// The order of simplices in the returned slice is not guaranteed to be fixed or sorted unless
  /// explicitly managed by internal operations (like those in `compute_homology`).
  ///
  /// # Arguments
  /// * `dimension`: The dimension of the simplices to retrieve.
  ///
  /// # Returns
  /// An `Option<&[Simplex]>` containing a slice of [`Simplex`] objects if simplices of that
  /// dimension exist, otherwise [`None`].
  pub fn simplices_by_dimension(&self, dimension: usize) -> Option<&[Simplex]> {
    self.simplices.get(&dimension).map(Vec::as_slice)
  }

  pub fn homology<F: Field + Copy>(&self, k: usize) -> Homology<Simplex, F> {
    // Get ordered bases for simplices of dimensions k-1, k, and k+1
    let k_simplices = self.simplices_by_dimension(k).map_or_else(Vec::new, |s| {
      let mut sorted = s.to_vec();
      sorted.sort_unstable();
      sorted
    });

    if k_simplices.is_empty() {
      return Homology::trivial(k);
    }

    // Special case for k=0: we need to handle connected components
    if k == 0 {
      // For H₀, we need to find connected components
      // Start with each vertex in its own component
      let mut components: Vec<Vec<Simplex>> = k_simplices.iter().map(|s| vec![s.clone()]).collect();

      // For each edge, merge the components of its vertices
      if let Some(edges) = self.simplices_by_dimension(1) {
        for edge in edges {
          let v0 = Simplex::new(0, vec![edge.vertices()[0]]);
          let v1 = Simplex::new(0, vec![edge.vertices()[1]]);

          // Find components containing v0 and v1
          let mut i0 = None;
          let mut i1 = None;
          for (i, comp) in components.iter().enumerate() {
            if comp.contains(&v0) {
              i0 = Some(i);
            }
            if comp.contains(&v1) {
              i1 = Some(i);
            }
          }

          // If vertices are in different components, merge them
          if let (Some(i0), Some(i1)) = (i0, i1) {
            if i0 != i1 {
              let comp1 = components.remove(i1);
              components[i0].extend(comp1);
            }
          }
        }
      }

      // Create one generator per connected component
      let homology_generators: Vec<Chain<Simplex, F>> = components
        .into_iter()
        .map(|comp| {
          let mut chain = Chain::new();
          for vertex in comp {
            chain = chain + Chain::from_item_and_coeff(vertex, F::one());
          }
          chain
        })
        .collect();

      return Homology {
        dimension: k,
        betti_number: homology_generators.len(),
        cycle_generators: k_simplices
          .iter()
          .map(|s| Chain::from_item_and_coeff(s.clone(), F::one()))
          .collect(),
        boundary_generators: Vec::new(),
        homology_generators,
      };
    }

    // For k > 0, proceed with normal homology computation
    let km1_simplices = self.simplices_by_dimension(k - 1).map_or_else(Vec::new, |s| {
      let mut sorted = s.to_vec();
      sorted.sort_unstable();
      sorted
    });

    let kp1_simplices = self.simplices_by_dimension(k + 1).map_or_else(Vec::new, |s| {
      let mut sorted = s.to_vec();
      sorted.sort_unstable();
      sorted
    });

    // Create basis chains for k-simplices
    let k_chain_basis: Vec<Chain<Simplex, F>> =
      k_simplices.iter().map(|s| Chain::from_item_and_coeff(s.clone(), F::one())).collect();

    // Compute boundary matrices
    let boundary_k = if km1_simplices.is_empty() {
      DynamicDenseMatrix::<F, RowMajor>::new()
    } else {
      get_boundary_matrix(&k_simplices, &km1_simplices)
    };

    let boundary_kp1 = if kp1_simplices.is_empty() {
      DynamicDenseMatrix::<F, RowMajor>::new()
    } else {
      get_boundary_matrix(&kp1_simplices, &k_simplices)
    };

    // Find cycles (kernel of ∂k) and boundaries (image of ∂k+1)
    let cycles = if km1_simplices.is_empty() {
      k_chain_basis
    } else {
      let kernel_vectors = boundary_k.kernel();
      kernel_vectors
        .into_iter()
        .map(|v| {
          let mut chain = Chain::new();
          for (i, &coeff) in v.components().iter().enumerate() {
            if !coeff.is_zero() {
              chain = chain + Chain::from_item_and_coeff(k_simplices[i].clone(), coeff);
            }
          }
          chain
        })
        .collect()
    };

    let boundaries = if kp1_simplices.is_empty() {
      Vec::new()
    } else {
      let image_vectors = boundary_kp1.image();
      image_vectors
        .into_iter()
        .map(|v| {
          let mut chain = Chain::new();
          for (i, &coeff) in v.components().iter().enumerate() {
            if !coeff.is_zero() {
              chain = chain + Chain::from_item_and_coeff(k_simplices[i].clone(), coeff);
            }
          }
          chain
        })
        .collect()
    };

    // Map k-simplices to their indices for converting chains to coefficient vectors.
    let k_simplex_to_idx: HashMap<Simplex, usize> =
      k_simplices.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    let num_k_simplices = k_simplices.len();

    // Convert cycles (chains) to coefficient vectors.
    // These are vectors in the basis of k_simplices.
    let cycle_vectors: Vec<_> = cycles // Using Vec<_> for DynamicVector<F>
      .iter()
      .map(|chain| chain.to_coeff_vector(&k_simplex_to_idx, num_k_simplices))
      .collect();

    // Convert boundaries (chains) to coefficient vectors.
    // These are also vectors in the basis of k_simplices.
    let boundary_vectors: Vec<_> = boundaries // Using Vec<_> for DynamicVector<F>
      .iter()
      .map(|chain| chain.to_coeff_vector(&k_simplex_to_idx, num_k_simplices))
      .collect();

    // Compute a basis for the quotient space Z_k / B_k using the imported function.
    // The result is a list of vectors (in k_simplices basis) representing homology generators.
    let quotient_basis_vectors = // Type is Vec<DynamicVector<F>>
        compute_quotient_basis(&boundary_vectors, &cycle_vectors);

    // Convert these basis vectors back into chains.
    let homology_generators: Vec<Chain<Simplex, F>> = quotient_basis_vectors
      .into_iter()
      .map(|v_coeff| {
        // v_coeff is a coefficient vector for a homology generator
        let mut generator_chain = Chain::new();
        for (idx_in_k_basis, &coeff) in v_coeff.components().iter().enumerate() {
          if !coeff.is_zero() {
            generator_chain = generator_chain
              + Chain::from_item_and_coeff(k_simplices[idx_in_k_basis].clone(), coeff);
          }
        }
        generator_chain
      })
      .collect();

    Homology {
      dimension: k,
      betti_number: homology_generators.len(),
      cycle_generators: cycles,
      boundary_generators: boundaries,
      homology_generators,
    }
  }
}

impl Topology for Simplex {
  type Space = SimplicialComplex;

  // TODO (autoparallel): Implement this. It  is the "star" of the simplex.
  fn neighborhood(&self) -> Vec<Self> { todo!() }

  fn boundary<R: Ring>(&self) -> Chain<Self, R> {
    if self.dimension == 0 {
      return Chain::new();
    }

    let mut boundary_chain_items = Vec::with_capacity(self.dimension + 1);
    let mut boundary_chain_coeffs = Vec::with_capacity(self.dimension + 1);

    // self.vertices are sorted: v_0, v_1, ..., v_k
    // Boundary is sum_{i=0 to k} (-1)^i * [v_0, ..., ^v_i, ..., v_k]
    for i in 0..=self.dimension {
      // i from 0 to k (k = self.dimension)
      let mut face_vertices = self.vertices.clone();
      face_vertices.remove(i); // Removes element at original index i (v_i)

      let face_simplex = Self::new(self.dimension - 1, face_vertices);
      boundary_chain_items.push(face_simplex);

      let coeff = if i % 2 == 0 {
        R::one()
      } else {
        -R::one() // Requires R: Neg
      };
      boundary_chain_coeffs.push(coeff);
    }
    Chain::from_items_and_coeffs(boundary_chain_items, boundary_chain_coeffs)
  }
}

/// Constructs the boundary matrix $\partial_k: C_k \to C_{k-1}$ for the $k$-th boundary operator.
///
/// The matrix columns are indexed by an ordered list of $k$-simplices (`ordered_k_simplices`),
/// and rows are indexed by an ordered list of $(k-1)$-simplices (`ordered_km1_simplices`).
/// The entry $(i,j)$ of the matrix is the coefficient of the $i$-th $(k-1)$-simplex in the
/// boundary of the $j$-th $k$-simplex.
///
/// The boundary of a $k$-simplex $\sigma_j = [v_0, \dots, v_k]$ is $\sum_{m=0}^{k} (-1)^m [v_0,
/// \dots, \hat{v}_m, \dots, v_k]$. The coefficient for $\sigma_i^{\prime}$ (the $i$-th
/// $(k-1)$-simplex) in $\partial \sigma_j$ is determined accordingly.
///
/// # Arguments
/// * `ordered_k_simplices`: A slice of [`Simplex`] objects, forming an ordered basis for the
///   $k$-chain group $C_k$.
/// * `ordered_km1_simplices`: A slice of [`Simplex`] objects, forming an ordered basis for the
///   $(k-1)$-chain group $C_{k-1}$.
///
/// # Type Parameters
/// * `F`: The coefficient field, must implement [`Field`] and `Copy`.
///
/// # Returns
/// A `Vec<Vec<F>>` representing the boundary matrix. The matrix will have
/// `ordered_km1_simplices.len()` rows and `ordered_k_simplices.len()` columns.
/// Returns an empty or specially-dimensioned matrix if either basis is empty.
pub fn get_boundary_matrix<F: Field + Copy>(
  ordered_k_simplices: &[Simplex],   // Basis for C_k (domain)
  ordered_km1_simplices: &[Simplex], // Basis for C_{k-1} (codomain)
) -> DynamicDenseMatrix<F, RowMajor> {
  if ordered_k_simplices.is_empty() || ordered_km1_simplices.is_empty() {
    return DynamicDenseMatrix::<F, RowMajor>::new();
  }

  let num_rows = ordered_km1_simplices.len();

  let mut matrix = DynamicDenseMatrix::<F, RowMajor>::new();

  // Create a map for quick lookup of (k-1)-simplex indices
  let km1_simplex_to_idx: HashMap<Simplex, usize> =
    ordered_km1_simplices.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

  for k_simplex in ordered_k_simplices {
    let boundary_chain = k_simplex.boundary();

    // Convert this boundary_chain into a column vector for the matrix.
    let col_vector = boundary_chain.to_coeff_vector(&km1_simplex_to_idx, num_rows);
    matrix.append_column(&col_vector);
  }
  matrix
}

#[cfg(test)]
mod tests {
  // TODO: Verify the homology generators are correct.

  use std::fmt::Debug;

  use harness_algebra::{algebras::boolean::Boolean, modular, prime_field, rings::Field};

  use super::*; // Make sure Mod7 is in scope here

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);
  // Helper trait bound alias for tests
  trait TestField: Field + Copy + Debug {}
  impl<T: Field + Copy + Debug> TestField for T {}

  #[test]
  fn test_simplex_faces() {
    let simplex = Simplex::new(2, vec![0, 1, 2]);
    let faces = simplex.faces();
    assert_eq!(faces.len(), 3);
    assert_eq!(faces[0].vertices(), &[0, 1]);
    assert_eq!(faces[1].vertices(), &[0, 2]);
    assert_eq!(faces[2].vertices(), &[1, 2]);
  }

  #[test]
  fn test_simplicial_complex() {
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(Simplex::new(2, vec![0, 1, 2]));
    assert_eq!(complex.simplices_by_dimension(2).unwrap().len(), 1);
    assert_eq!(complex.simplices_by_dimension(1).unwrap().len(), 3);
    assert_eq!(complex.simplices_by_dimension(0).unwrap().len(), 3);
  }

  #[test]
  fn test_chain_addition_disjoint() {
    // Create two chains with different simplices
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![1, 2]);

    let chain1 = Chain::from_item_and_coeff(simplex1, 1_i32);
    let chain2 = Chain::from_item_and_coeff(simplex2, 2_i32);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items[0].vertices(), &[0, 1]);
    assert_eq!(result.items[1].vertices(), &[1, 2]);
    assert_eq!(result.coefficients[0], 1);
    assert_eq!(result.coefficients[1], 2);
  }

  #[test]
  fn test_chain_addition_same_simplex() {
    // Create two chains with the same simplex
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);

    let chain1 = Chain::from_item_and_coeff(simplex1, 2);
    let chain2 = Chain::from_item_and_coeff(simplex2, 3);

    let result = chain1 + chain2;

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.coefficients.len(), 1);
    assert_eq!(result.items[0].vertices(), &[0, 1]);
    assert_eq!(result.coefficients[0], 5); // 2 + 3 = 5
  }

  #[test]
  fn test_chain_addition_canceling_coefficients() {
    // Create two chains with the same simplex but opposite coefficients
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);

    let chain1 = Chain::from_item_and_coeff(simplex1, 2);
    let chain2 = Chain::from_item_and_coeff(simplex2, -2);

    let result = chain1 + chain2;

    // The result should be empty since the coefficients cancel out
    assert_eq!(result.items.len(), 0);
    assert_eq!(result.coefficients.len(), 0);
  }

  #[test]
  fn test_chain_boundary_edge() {
    // The boundary of an edge is its two vertices with opposite signs
    let edge = Simplex::new(1, vec![0, 1]);
    let chain = Chain::from_item_and_coeff(edge, 1);

    let boundary = chain.boundary();

    // Should have two 0-simplices (vertices)
    assert_eq!(boundary.items.len(), 2);
    assert_eq!(boundary.coefficients.len(), 2);

    // Verify the vertices
    assert!(boundary.items.iter().any(|s| s.vertices().contains(&0)));
    assert!(boundary.items.iter().any(|s| s.vertices().contains(&1)));
    assert!(boundary.items.len() == 2);

    // Verify opposite signs (exact sign depends on your implementation)
    assert_eq!(boundary.coefficients[0] + boundary.coefficients[1], 0);
  }

  #[test]
  fn test_chain_boundary_triangle() {
    // The boundary of a triangle is its three edges
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let chain = Chain::from_item_and_coeff(triangle, 1);

    let boundary = chain.boundary();

    // Should have three 1-simplices (edges)
    assert_eq!(boundary.items.len(), 3);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.items.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_boundary_squared_is_zero() {
    // Verify that ∂² = 0 for a triangle
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let chain = Chain::from_item_and_coeff(triangle, 1);

    let boundary = chain.boundary();
    let boundary_squared = boundary.boundary();

    // Boundary of boundary should be empty (∂² = 0)
    assert_eq!(boundary_squared.items.len(), 0);
    assert_eq!(boundary_squared.coefficients.len(), 0);
  }

  #[test]
  fn test_complex_chain_operations() {
    // Create a 2-chain with two triangles sharing an edge
    let triangle1 = Simplex::new(2, vec![0, 1, 2]);
    let triangle2 = Simplex::new(2, vec![1, 2, 3]);

    let chain1 = Chain::from_item_and_coeff(triangle1, 1);
    let chain2 = Chain::from_item_and_coeff(triangle2, -1);

    let combined_chain = chain1 + chain2;
    let boundary = combined_chain.boundary();

    // The boundary should have 4 edges (the shared edge [1,2] cancels out)
    assert_eq!(boundary.items.len(), 4);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.items.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 3]));
    assert!(edge_vertices.contains(&vec![2, 3]));

    // The shared edge [1,2] should not be present because its coefficients cancel
    assert!(!edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_simplices_by_dimension_basic() {
    let mut complex = SimplicialComplex::new();

    // Adding a 2-simplex will also add its 1-simplex faces and 0-simplex vertices
    // due to the behavior of join_simplex.
    let s2_v012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_simplex(s2_v012.clone());

    // Expected 0-simplices (vertices)
    let s0_v0 = Simplex::new(0, vec![0]);
    let s0_v1 = Simplex::new(0, vec![1]);
    let s0_v2 = Simplex::new(0, vec![2]);

    // Expected 1-simplices (edges)
    let s1_v01 = Simplex::new(1, vec![0, 1]); // face of s2_v012
    let s1_v02 = Simplex::new(1, vec![0, 2]); // face of s2_v012
    let s1_v12 = Simplex::new(1, vec![1, 2]); // face of s2_v012

    // Check dimension 2
    let dim2_simplices = complex.simplices_by_dimension(2).expect("Dim 2 should exist");
    assert_eq!(dim2_simplices.len(), 1, "Should be one 2-simplex");
    assert!(dim2_simplices.contains(&s2_v012), "Missing 2-simplex [0,1,2]");

    // Check dimension 1
    let dim1_simplices = complex.simplices_by_dimension(1).expect("Dim 1 should exist");
    assert_eq!(dim1_simplices.len(), 3, "Should be three 1-simplices");
    assert!(dim1_simplices.contains(&s1_v01), "Missing 1-simplex [0,1]");
    assert!(dim1_simplices.contains(&s1_v02), "Missing 1-simplex [0,2]");
    assert!(dim1_simplices.contains(&s1_v12), "Missing 1-simplex [1,2]");

    // Check dimension 0
    let dim0_simplices = complex.simplices_by_dimension(0).expect("Dim 0 should exist");
    assert_eq!(dim0_simplices.len(), 3, "Should be three 0-simplices");
    assert!(dim0_simplices.contains(&s0_v0), "Missing 0-simplex [0]");
    assert!(dim0_simplices.contains(&s0_v1), "Missing 0-simplex [1]");
    assert!(dim0_simplices.contains(&s0_v2), "Missing 0-simplex [2]");
  }

  fn test_homology_point_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let p0 = Simplex::new(0, vec![0]);
    complex.join_simplex(p0.clone());

    // H_0
    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");
    let expected_gen_h0 = Chain::from_item_and_coeff(p0, F::one());
    assert!(
      h0.homology_generators.contains(&expected_gen_h0),
      "H0: Generator mismatch for field {:?}",
      std::any::type_name::<F>()
    );

    // H_1
    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Should have no generators for field {:?}",
      std::any::type_name::<F>()
    );

    // H_2
    let h2 = complex.homology::<F>(2);
    assert_eq!(
      h2.betti_number,
      0,
      "H2: Betti number for a point should be 0 for field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_point_all_fields() {
    test_homology_point_generic::<Boolean>();
    test_homology_point_generic::<Mod7>();
  }

  fn test_homology_edge_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let edge01 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(edge01);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti for an edge");
    assert_eq!(h0.homology_generators.len(), 1, "H0: One generator");
    let p0 = Simplex::new(0, vec![0]);
    let expected_gen_h0 = Chain::from_item_and_coeff(p0, F::one());
    assert!(
      h0.homology_generators.contains(&expected_gen_h0),
      "H0: Generator for edge field {:?}",
      std::any::type_name::<F>()
    );

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti for an edge");
    assert!(
      h1.homology_generators.is_empty(),
      "H1: No generators for edge field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_edge_all_fields() {
    test_homology_edge_generic::<Boolean>();
    test_homology_edge_generic::<Mod7>();
  }

  fn test_homology_two_disjoint_points_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let p0_s = Simplex::new(0, vec![0]);
    let p1_s = Simplex::new(0, vec![1]);
    complex.join_simplex(p0_s.clone());
    complex.join_simplex(p1_s.clone());

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 2, "H0: Betti for two points");
    assert_eq!(h0.homology_generators.len(), 2, "H0: Two generators");
    let expected_gen1_h0 = Chain::from_item_and_coeff(p0_s, F::one());
    let expected_gen2_h0 = Chain::from_item_and_coeff(p1_s, F::one());
    assert!(
      h0.homology_generators.contains(&expected_gen1_h0),
      "H0: Gen [0] missing field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      h0.homology_generators.contains(&expected_gen2_h0),
      "H0: Gen [1] missing field {:?}",
      std::any::type_name::<F>()
    );

    let h1 = complex.homology::<F>(1);
    assert_eq!(
      h1.betti_number,
      0,
      "H1: Betti for two points field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_two_disjoint_points_all_fields() {
    test_homology_two_disjoint_points_generic::<Boolean>();
    test_homology_two_disjoint_points_generic::<Mod7>();
  }

  fn test_homology_filled_triangle_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let triangle012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_simplex(triangle012);

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for triangle field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(
      h1.betti_number,
      0,
      "H1: Betti for filled triangle field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      h1.homology_generators.is_empty(),
      "H1: No 1D generators field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(
      h2.betti_number,
      0,
      "H2: Betti for filled triangle field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      h2.homology_generators.is_empty(),
      "H2: No 2D generators field {:?}",
      std::any::type_name::<F>()
    );
  }

  #[test]
  fn test_homology_filled_triangle_all_fields() {
    test_homology_filled_triangle_generic::<Boolean>();
    test_homology_filled_triangle_generic::<Mod7>();
  }

  fn test_homology_circle_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let s01 = Simplex::new(1, vec![0, 1]);
    let s12 = Simplex::new(1, vec![1, 2]);
    let s02 = Simplex::new(1, vec![0, 2]);

    complex.join_simplex(s01.clone());
    complex.join_simplex(s12.clone());
    complex.join_simplex(s02.clone());

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for circle field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 1, "H1: Betti for circle field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h1.homology_generators.len(),
      1,
      "H1: One 1D generator field {:?}",
      std::any::type_name::<F>()
    );

    let generator_h1 = h1.homology_generators[0].clone();
    assert_eq!(
      generator_h1.items.len(),
      3,
      "H1: Circle generator 3 simplices field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.items.contains(&s01),
      "H1: Gen missing s01 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.items.contains(&s12),
      "H1: Gen missing s12 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.items.contains(&s02),
      "H1: Gen missing s02 field {:?}",
      std::any::type_name::<F>()
    );

    // Check that all coefficients of the generator are F::one() or -F::one().
    assert!(
      !generator_h1.coefficients.is_empty(),
      "H1: Generator should have coefficients for field {:?}",
      std::any::type_name::<F>()
    );
    let one = F::one();
    let neg_one = -F::one();

    for coeff in &generator_h1.coefficients {
      assert!(
        *coeff == one || *coeff == neg_one,
        "H1: Each coefficient in the generator should be F::one() or -F::one(). Found: {:?} for \
         field {:?}",
        *coeff,
        std::any::type_name::<F>()
      );
    }

    assert!(
      generator_h1.boundary().items.is_empty(),
      "H1: Generator boundary zero field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for circle field {:?}", std::any::type_name::<F>());
  }

  #[test]
  fn test_homology_circle_all_fields() {
    test_homology_circle_generic::<Boolean>();
    test_homology_circle_generic::<Mod7>();
  }

  fn test_homology_sphere_surface_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let f012 = Simplex::new(2, vec![0, 1, 2]);
    let f013 = Simplex::new(2, vec![0, 1, 3]);
    let f023 = Simplex::new(2, vec![0, 2, 3]);
    let f123 = Simplex::new(2, vec![1, 2, 3]);

    complex.join_simplex(f012.clone());
    complex.join_simplex(f013.clone());
    complex.join_simplex(f023.clone());
    complex.join_simplex(f123.clone());

    let h0 = complex.homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti sphere field {:?}", std::any::type_name::<F>());

    let h1 = complex.homology::<F>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti sphere field {:?}", std::any::type_name::<F>());
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Generators sphere field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.homology::<F>(2);
    assert_eq!(h2.betti_number, 1, "H2: Betti sphere field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h2.homology_generators.len(),
      1,
      "H2: One 2D generator field {:?}",
      std::any::type_name::<F>()
    );

    let generator_h2 = h2.homology_generators[0].clone();
    assert_eq!(
      generator_h2.items.len(),
      4,
      "H2: Sphere generator 4 faces field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.items.contains(&f012),
      "H2: Gen missing f012 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.items.contains(&f013),
      "H2: Gen missing f013 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.items.contains(&f023),
      "H2: Gen missing f023 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.items.contains(&f123),
      "H2: Gen missing f123 field {:?}",
      std::any::type_name::<F>()
    );

    // Check that all coefficients of the generator are F::one() or -F::one().
    assert!(
      !generator_h2.coefficients.is_empty(),
      "H2: Generator should have coefficients for field {:?}",
      std::any::type_name::<F>()
    );
    let one = F::one();
    let neg_one = -F::one();

    for coeff in &generator_h2.coefficients {
      assert!(
        *coeff == one || *coeff == neg_one,
        "H2: Each coefficient in the generator should be F::one() or -F::one(). Found: {:?} for \
         field {:?}",
        *coeff,
        std::any::type_name::<F>()
      );
    }

    assert!(
      generator_h2.boundary().items.is_empty(),
      "H2: Generator boundary zero field {:?}",
      std::any::type_name::<F>()
    );

    let h3 = complex.homology::<F>(3);
    assert_eq!(h3.betti_number, 0, "H3: Betti sphere field {:?}", std::any::type_name::<F>());
  }

  #[test]
  fn test_homology_sphere_surface_all_fields() {
    test_homology_sphere_surface_generic::<Boolean>();
    test_homology_sphere_surface_generic::<Mod7>();
  }
}
