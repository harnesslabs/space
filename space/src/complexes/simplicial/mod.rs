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
//! use harness_space::simplicial::{Simplex, SimplicialComplex}; // For Z/2Z coefficients
//!
//! // Create a simplicial complex representing a hollow triangle (cycle C3)
//! let mut complex = SimplicialComplex::new();
//! complex.join_simplex(Simplex::new(1, vec![0, 1])); // Edge (0,1)
//! complex.join_simplex(Simplex::new(1, vec![1, 2])); // Edge (1,2)
//! complex.join_simplex(Simplex::new(1, vec![2, 0])); // Edge (2,0)
//!
//! // Compute H_0 and H_1 with Z/2Z coefficients
//! let h0 = complex.compute_homology::<Boolean>(0);
//! let h1 = complex.compute_homology::<Boolean>(1);
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

use std::collections::{HashMap, HashSet};

use harness_algebra::tensors::dynamic::matrix::{DynamicDenseMatrix, RowMajor};
use itertools::Itertools;

use super::*;
use crate::{
  definitions::Topology,
  homology::{Chain, Homology},
};

#[cfg(test)] mod tests;

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

    // Create matrix for quotient space computation
    let mut quotient_matrix = DynamicDenseMatrix::<F, RowMajor>::new();
    let k_simplex_to_idx: HashMap<Simplex, usize> =
      k_simplices.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();

    // Add boundary vectors first, then cycle vectors
    for boundary in &boundaries {
      quotient_matrix
        .append_column(&boundary.to_coeff_vector(&k_simplex_to_idx, k_simplices.len()));
    }
    for cycle in &cycles {
      quotient_matrix.append_column(&cycle.to_coeff_vector(&k_simplex_to_idx, k_simplices.len()));
    }

    // Find homology generators
    let homology_generators = if quotient_matrix.num_cols() == 0 {
      cycles.clone()
    } else {
      let rref_result = quotient_matrix.row_echelon_form();
      let num_boundaries = boundaries.len();

      // Get the pivot columns from the RREF
      let pivot_cols: HashSet<usize> = rref_result.pivots.iter().map(|p| p.col).collect();

      // For each cycle, check if it's a boundary
      cycles
        .iter()
        .enumerate()
        .filter_map(|(i, cycle)| {
          let cycle_col = num_boundaries + i;
          if pivot_cols.contains(&cycle_col) {
            Some(cycle.clone())
          } else {
            None
          }
        })
        .collect()
    };

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
