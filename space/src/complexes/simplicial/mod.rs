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

use std::collections::HashMap;

use harness_algebra::tensors::dynamic::{
  matrix::{DynamicDenseMatrix, RowMajor},
  vector::DynamicVector,
};
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

  // // TODO (autoparallel): This is a bit of a mess. We should clean it up.
  #[allow(clippy::too_many_lines)]
  /// Computes the $k$-th homology group $H_k(K; F)$ of the simplicial complex $K$
  /// with coefficients in a specified field $F$.
  ///
  /// Homology groups are algebraic invariants that capture information about the "holes"
  /// in a topological space. For example:
  /// - $H_0$ describes path-connected components (rank is the number of components).
  /// - $H_1$ describes 1-dimensional holes (like loops or tunnels).
  /// - $H_2$ describes 2-dimensional voids (like cavities).
  ///
  /// The computation involves:
  /// 1. Identifying bases for $k$-simplices ($S_k$), $(k-1)$-simplices ($S_{k-1}$), and
  ///    $(k+1)$-simplices ($S_{k+1}$).
  /// 2. Constructing boundary matrices for $\partial_k: C_k \to C_{k-1}$ and $\partial_{k+1}:
  ///    C_{k+1} \to C_k$.
  /// 3. Using [`DynamicDenseMatrix::row_echelon_form`] to find the rank and bases for the kernel
  ///    (cycles $Z_k = \ker \partial_k$) and image (boundaries $B_k = \text{im } \partial_{k+1}$).
  /// 4. The homology group $H_k = Z_k / B_k$. The Betti number ($b_k$) is $\text{rank}(H_k) =
  ///    \dim(Z_k) - \dim(B_k)$.
  ///
  /// # Type Parameters
  /// * `F`: The coefficient field. Must implement [`Field`] and [`Copy`]. Common choices include
  ///   [`f64`], [`Boolean`](harness_algebra::algebras::boolean::Boolean) (for
  ///   $\mathbb{Z}/2\mathbb{Z}$), or modular fields like $\mathbb{Z}/p\mathbb{Z}$ for prime $p$.
  ///
  /// # Arguments
  /// * `k`: The dimension of the homology group to compute (e.g., 0 for $H_0$, 1 for $H_1$).
  ///
  /// # Returns
  /// A [`HomologyGroup<F>`] struct containing the dimension, Betti number, and generators for
  /// cycles, boundaries, and the homology group itself.
  /// Returns a trivial homology group if there are no $k$-simplices or if other conditions lead to
  /// a trivial group.
  pub fn homology<F: Field + Copy>(&self, k: usize) -> Homology<Simplex, F> {
    // TODO: Let's think about this, we always need to find the image of \partial_{k+1} and the
    // kernel of \partial_k. However, the kernel of \partial_k when k=0 is all the vertices. So we
    // should make this kind of conditional work nicely here.
    // TODO: We should also make use of the matrix lib to do this, so we should be able to determine
    // image and kernel of matrix. We should also just have a simple way to create a vector from
    // chains (which we can do by sorting).
    if k == 0 {
      let mut vertices = match self.simplices_by_dimension(0) {
        Some(v) => v.to_vec(),
        None => return Homology::trivial(0),
      };
      if vertices.is_empty() {
        return Homology::trivial(0);
      }
      vertices.sort_unstable();

      let mut adj: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
      let vertex_to_idx: std::collections::HashMap<Simplex, usize> =
        vertices.iter().enumerate().map(|(i, v_s)| (v_s.clone(), i)).collect();

      if let Some(edges) = self.simplices_by_dimension(1) {
        for edge_s in edges {
          if edge_s.vertices().len() == 2 {
            let v0_s = Simplex::new(0, vec![edge_s.vertices()[0]]);
            let v1_s = Simplex::new(0, vec![edge_s.vertices()[1]]);
            if let (Some(&idx0), Some(&idx1)) = (vertex_to_idx.get(&v0_s), vertex_to_idx.get(&v1_s))
            {
              adj[idx0].push(idx1);
              adj[idx1].push(idx0);
            }
          }
        }
      }
      let mut visited = vec![false; vertices.len()];
      let mut components = 0;
      let mut h0_generators = Vec::new();
      for i in 0..vertices.len() {
        if !visited[i] {
          components += 1;
          h0_generators.push(Chain::from_item_and_coeff(vertices[i].clone(), <F as One>::one()));
          let mut q = std::collections::VecDeque::new();
          q.push_back(i);
          visited[i] = true;
          while let Some(u) = q.pop_front() {
            for &v_idx in &adj[u] {
              if !visited[v_idx] {
                visited[v_idx] = true;
                q.push_back(v_idx);
              }
            }
          }
        }
      }
      let z0_gens: Vec<Chain<Simplex, F>> =
        vertices.iter().map(|v| Chain::from_item_and_coeff(v.clone(), F::one())).collect();
      let s1_for_b0 = self.simplices_by_dimension(1).map_or_else(Vec::new, |s| {
        let mut sorted_s = s.to_vec();
        sorted_s.sort_unstable();
        sorted_s
      });
      let b0_gens = if s1_for_b0.is_empty() || vertices.is_empty() {
        Vec::new()
      } else {
        let mut mat_d1: DynamicDenseMatrix<F, RowMajor> =
          get_boundary_matrix(&s1_for_b0, &vertices);
        let mat_d1_result = mat_d1.row_echelon_form();
        let d_s1_chains: Vec<Chain<Simplex, F>> = s1_for_b0
          .iter()
          .map(|s| Chain::from_item_and_coeff(s.clone(), F::one()).boundary())
          .collect();
        image_basis_from_row_echelon(
          &d_s1_chains,
          &mat_d1_result.pivots.iter().map(|p| p.col).collect::<Vec<_>>(),
        )
      };
      return Homology {
        dimension:           0,
        betti_number:        components,
        cycle_generators:    z0_gens,
        boundary_generators: b0_gens,
        homology_generators: h0_generators,
      };
    }

    let s_k = match self.simplices_by_dimension(k) {
      Some(s) if !s.is_empty() => {
        let mut sorted_s = s.to_vec();
        sorted_s.sort_unstable();
        sorted_s
      },
      _ => return Homology::trivial(k),
    };
    let s_km1 = self.simplices_by_dimension(k - 1).map_or_else(Vec::new, |s| {
      let mut ss = s.to_vec();
      ss.sort_unstable();
      ss
    });
    let s_kp1 = self.simplices_by_dimension(k + 1).map_or_else(Vec::new, |s| {
      let mut ss = s.to_vec();
      ss.sort_unstable();
      ss
    });
    let ck_basis: Vec<Chain<Simplex, F>> =
      s_k.iter().map(|s| Chain::from_item_and_coeff(s.clone(), <F as One>::one())).collect();

    let b_k_gens = if s_kp1.is_empty() || s_k.is_empty() {
      Vec::new()
    } else {
      let mut mat_dkp1: DynamicDenseMatrix<F, RowMajor> = get_boundary_matrix(&s_kp1, &s_k);
      let mat_dkp1_result = mat_dkp1.row_echelon_form();
      let d_skp1_chains: Vec<Chain<Simplex, F>> = s_kp1
        .iter()
        .map(|s| Chain::from_item_and_coeff(s.clone(), <F as One>::one()).boundary())
        .collect();
      image_basis_from_row_echelon(
        &d_skp1_chains,
        &mat_dkp1_result.pivots.iter().map(|p| p.col).collect::<Vec<_>>(),
      )
    };

    let z_k_gens = if s_k.is_empty() {
      Vec::new()
    } else if s_km1.is_empty() {
      ck_basis
    } else {
      let mut mat_dk = get_boundary_matrix(&s_k, &s_km1);
      let mat_dk_result = mat_dk.row_echelon_form();
      kernel_basis_from_row_echelon(
        &mat_dk,
        &ck_basis,
        &mat_dk_result.pivots.iter().map(|p| p.col).collect::<Vec<_>>(),
      )
    };

    if s_k.is_empty() {
      return Homology::trivial(k);
    }
    let sk_map: std::collections::HashMap<Simplex, usize> =
      s_k.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    let mut quot_mat_cols = Vec::new();
    for ch in &b_k_gens {
      quot_mat_cols.push(ch.to_coeff_vector(&sk_map, s_k.len()));
    }
    let num_b = quot_mat_cols.len();
    for ch in &z_k_gens {
      quot_mat_cols.push(ch.to_coeff_vector(&sk_map, s_k.len()));
    }

    if quot_mat_cols.is_empty() {
      return Homology {
        dimension:           k,
        betti_number:        z_k_gens.len(),
        cycle_generators:    z_k_gens.clone(),
        boundary_generators: b_k_gens,
        homology_generators: z_k_gens,
      };
    }

    let mut q_mat = DynamicDenseMatrix::<F, RowMajor>::new();
    if !s_k.is_empty() {
      for c in quot_mat_cols {
        q_mat.append_column(&c);
      }
    }

    let p_cols_q = q_mat.row_echelon_form().pivots.iter().map(|p| p.col).collect::<Vec<_>>();
    let mut h_k_gens = Vec::new();
    for &p_idx in &p_cols_q {
      if p_idx >= num_b {
        h_k_gens.push(z_k_gens[p_idx - num_b].clone());
      }
    }

    Homology {
      dimension:           k,
      betti_number:        h_k_gens.len(),
      cycle_generators:    z_k_gens,
      boundary_generators: b_k_gens,
      homology_generators: h_k_gens,
    }
  }
}

// TODO: Permutation should probably be moved into algebra crate for permutation groups.
/// Represents whether a permutation of vertices is odd or even.
/// This is used, for example, in defining the orientation of a simplex, although
/// the `Chain::eq` method's use of it might need review for standard chain equality.
#[derive(Debug, PartialEq, Eq)]
pub enum Permutation {
  /// An odd permutation (e.g., requires an odd number of transpositions to reach sorted order).
  Odd,
  /// An even permutation (e.g., requires an even number of transpositions to reach sorted order).
  Even,
}

/// Computes the sign (even or odd) of a permutation by counting the number of inversions.
/// An inversion is a pair of elements `(item[i], item[j])` such that `i < j` but `item[i] >
/// item[j]`.
///
/// # Arguments
/// * `item`: A slice of ordered items (e.g., vertex indices of a simplex before sorting).
///
/// # Returns
/// * `Permutation::Even` if the number of inversions is even.
/// * `Permutation::Odd` if the number of inversions is odd.
pub fn permutation_sign<V: Ord>(item: &[V]) -> Permutation {
  let mut count = 0;
  for i in 0..item.len() {
    for j in i + 1..item.len() {
      if item[i] > item[j] {
        count += 1;
      }
    }
  }
  if count % 2 == 0 {
    Permutation::Even
  } else {
    Permutation::Odd
  }
}

impl Permutation {
  pub fn into_ring<R: Ring>(self) -> R {
    match self {
      Permutation::Even => R::one(),
      Permutation::Odd => -R::one(),
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
    let mut boundary = Chain::new();
    for face in self.faces() {
      let coeff = permutation_sign(face.vertices()).into_ring::<R>();
      boundary = boundary + Chain::from_items_and_coeffs(vec![face], vec![coeff]);
    }
    boundary
  }
}

// /// Extracts a basis for the kernel (null space) of a linear transformation represented by
// /// `row_echelon_matrix`.
// ///
// /// The matrix $A$ (for which `row_echelon_matrix` is its RREF) maps a vector space $V$ (spanned
// by /// `column_basis_elements`) to a vector space $W$. This function finds a basis for $\ker A =
// \{ v /// \in V \mid Av = 0 \}$.
// ///
// /// The method involves identifying free variables from the RREF and expressing pivot variables
// in /// terms of them. Each free variable gives rise to a kernel basis vector.
// ///
// /// # Arguments
// /// * `row_echelon_matrix`: The matrix in reduced row echelon form (RREF).
// /// * `column_basis_elements`: A slice of [`Chain<F>`]s representing the basis for the domain
// $V$. ///   The $j$-th chain corresponds to the $j$-th column of the original matrix before RREF.
// The ///   length of this slice must match the number of columns in `row_echelon_matrix`.
// /// * `pivot_cols`: A slice of `usize` containing the column indices of the pivot elements in the
// ///   RREF.
// ///
// /// # Type Parameters
// /// * `F`: The coefficient field, must implement [`Field`] and `Copy`.
// ///
// /// # Returns
// /// A `Vec<Chain<F>>` where each chain is a generator for the kernel.
// ///
// /// # Panics
// /// * If `column_basis_elements.len()` does not match the number of columns in
// `row_echelon_matrix`. ///
// /// # Note
// /// The construction of `gen_chain` currently involves element-wise multiplication of
// coefficients /// within the `single_term_chain`. This assumes that `Chain::add` correctly
// combines these scaled /// chains. A more robust `Chain` API might include explicit scalar
// multiplication. pub fn kernel_basis_from_row_echelon<F: Field + Copy>(
//   row_echelon_matrix: &DynamicDenseMatrix<F, RowMajor>,
//   column_basis_elements: &[Chain<F>],
//   pivot_cols: &[usize],
// ) -> Vec<Chain<F>> {
//   if row_echelon_matrix.num_rows() == 0 {
//     if column_basis_elements.is_empty() {
//       return Vec::new();
//     }
//     // If it's a zero map from non-trivial C_n to C_m (m > 0), or to C_0,
//     // then all of C_n is kernel.
//     // This case should be handled if num_rows (effective rank) is 0 but num_cols > 0.
//     // The current logic for free variables should correctly make all columns free if rank is 0.
//     let is_zero_matrix_effectively = pivot_cols.is_empty();
//     if is_zero_matrix_effectively && !column_basis_elements.is_empty() {
//       return column_basis_elements.to_vec();
//     }
//     if column_basis_elements.is_empty() {
//       // Explicitly handle if no columns to form basis from
//       return Vec::new();
//     }
//     // Fallback if matrix had rows but all were zero, or other edge cases
//     // This part of the original condition might be too simple.
//     // Let the main loop handle it: if all columns are free, it will generate all basis elements.
//   }

//   let num_cols = if row_echelon_matrix.num_rows() == 0 { 0 } else { row_echelon_matrix.num_cols()
// };   if num_cols == 0 && column_basis_elements.is_empty() {
//     return Vec::new();
//   }
//   // If num_cols is 0 but column_basis_elements is not, it's an inconsistency.
//   // If row_echelon_matrix is empty but column_basis_elements is not, it implies a map to C_0.
//   // In this case, all columns are free. num_rows would be 0.

//   assert_eq!(
//     num_cols,
//     column_basis_elements.len(),
//     "Number of columns in matrix must match number of column basis elements"
//   );

//   let mut kernel_generators = Vec::new();
//   let mut current_pivot_idx = 0;

//   for j_col in 0..num_cols {
//     // Iterate through columns (potential free variables)
//     if current_pivot_idx < pivot_cols.len() && pivot_cols[current_pivot_idx] == j_col {
//       // This is a pivot column
//       current_pivot_idx += 1;
//     } else {
//       // This is a free variable column j_col
//       let mut generator_coeffs = vec![<F as Zero>::zero(); num_cols];
//       generator_coeffs[j_col] = <F as One>::one(); // Set free variable x_{j_col} to 1

//       // Solve for pivot variables in terms of this free variable
//       // For each pivot row r, associated with a pivot column p_c = pivot_cols[r_idx]:
//       //   x_{p_c} * 1 + M[r][j_col] * x_{j_col} = 0  (other free vars are 0, other pivots are 0
//       // in this equation)   x_{p_c} = -M[r][j_col] * x_{j_col}
//       // Since x_{j_col} = 1, then x_{p_c} = -M[r][j_col]

//       // Iterate through the pivot rows of the row_echelon_matrix
//       // The number of actual pivot rows is pivot_cols.len() (which is the rank)
//       // `row_echelon_matrix` still has all original rows, but non-pivot rows are zero or become
//       // zero relevant to pivots.

//       (0..pivot_cols.len()).for_each(|r_idx| {
//         // Iterate over actual pivot rows by their index in pivot_cols
//         let pivot_row_actual_index = r_idx; // In reduced row echelon form, pivot r is in row r.
//                                             // The way `row_gaussian_elimination` is written, it
// processes row by row.                                             // The pivot `matrix[r][lead]`
// means `pivot_cols[r_idx]` is that `lead`.         let pivot_col_for_this_row = pivot_cols[r_idx];

//         // We need to ensure row_echelon_matrix[pivot_row_actual_index] is the correct row.
//         // The RRE matrix has its pivots matrix[r_idx][pivot_cols[r_idx]] = 1.
//         if !row_echelon_matrix.get_component(pivot_row_actual_index, j_col).is_zero() {
//           generator_coeffs[pivot_col_for_this_row] =
//             -*row_echelon_matrix.get_component(pivot_row_actual_index, j_col);
//         }
//       });

//       let mut gen_chain = Chain::new();
//       for (idx, coeff) in generator_coeffs.iter().enumerate() {
//         if !coeff.is_zero() {
//           // This creates a chain with one simplex and the calculated coefficient.
//           // Then adds it to gen_chain. This is not quite right.
//           // We need to sum up (coeff_i * basis_chain_i).
//           // gen_chain = gen_chain + (coeff.clone() * column_basis_elements[idx].clone());
//           // Chain does not currently support multiplication by scalar from the left.
//           // For now, assume coeff is 1 or -1, or for Z2, just 1.
//           // If coeff is F::one(), add column_basis_elements[idx]
//           // If coeff is -F::one(), subtract column_basis_elements[idx] (add its negation)
//           // This logic needs Chain to support scalar multiplication or careful construction.

//           // Current Chain::add combines like terms. We are building a new chain from a linear
//           // combination. Let's assume column_basis_elements are single simplices with
//           // coeff 1.
//           let mut single_term_chain = column_basis_elements[idx].clone();
//           // We need to scale single_term_chain by *coeff.
//           // A simple way for now, if we assume Field allows it:
//           for c in &mut single_term_chain.coefficients {
//             *c *= *coeff; // Assumes coeff is F, c is F.
//           }
//           if !coeff.is_zero() {
//             // Re-check after potential multiplication if coeff was complex.
//             gen_chain = gen_chain + single_term_chain;
//           }
//         }
//       }
//       if !gen_chain.simplices.is_empty() {
//         kernel_generators.push(gen_chain);
//       }
//     }
//   }
//   kernel_generators
// }

// /// Extracts a basis for the image (column space) of a linear transformation represented by a
// /// matrix.
// ///
// /// If $A$ is the original matrix (before RREF), and `column_basis_elements` are the chains
// /// corresponding to the columns of $A$, then the chains corresponding to the pivot columns of
// $A$ /// (identified from its RREF) form a basis for $\text{im } A$.
// ///
// /// # Arguments
// /// * `_row_echelon_matrix`: The matrix in RREF. Not strictly used if `pivot_cols` is accurate,
// but ///   kept for context.
// /// * `column_basis_elements`: A slice of [`Chain<F>`]s representing the basis for the domain,
// ///   corresponding to columns of the original matrix.
// /// * `pivot_cols`: A slice of `usize` containing the column indices of the pivot elements in the
// ///   RREF of the original matrix.
// ///
// /// # Type Parameters
// /// * `F`: The coefficient field, must implement [`Field`] and `Copy`.
// ///
// /// # Returns
// /// A `Vec<Chain<F>>` where each chain is a generator for the image.
// pub fn image_basis_from_row_echelon<F: Field + Copy>(
//   column_basis_elements: &[Chain<F>],
//   pivot_cols: &[usize],
// ) -> Vec<Chain<F>> {
//   let mut image_generators = Vec::new();
//   for &pivot_col_idx in pivot_cols {
//     if pivot_col_idx < column_basis_elements.len() {
//       image_generators.push(column_basis_elements[pivot_col_idx].clone());
//     }
//   }
//   image_generators
// }

// // Helper function to convert a Chain to a coefficient vector relative to a basis of simplices.
// // This is an internal utility function used during the construction of boundary matrices.
// // It projects a chain onto a vector representation given a specific ordered basis of simplices.
// fn chain_to_coeff_vector<F: Field + Copy>(
//   chain: &Chain<F>,
//   basis_simplices_map: &HashMap<Simplex, usize>, // Map simplex to its index in the basis
//   basis_size: usize,
// ) -> DynamicVector<F> {
//   let mut vector = DynamicVector::<F>::new(vec![<F as Zero>::zero(); basis_size]);
//   for (i, simplex) in chain.simplices.iter().enumerate() {
//     if let Some(&idx) = basis_simplices_map.get(simplex) {
//       if !chain.coefficients[i].is_zero() {
//         vector.set_component(idx, chain.coefficients[i]);
//       }
//     }
//   }
//   vector
// }

// /// Constructs the boundary matrix $\partial_k: C_k \to C_{k-1}$ for the $k$-th boundary
// operator. ///
// /// The matrix columns are indexed by an ordered list of $k$-simplices (`ordered_k_simplices`),
// /// and rows are indexed by an ordered list of $(k-1)$-simplices (`ordered_km1_simplices`).
// /// The entry $(i,j)$ of the matrix is the coefficient of the $i$-th $(k-1)$-simplex in the
// boundary /// of the $j$-th $k$-simplex.
// ///
// /// The boundary of a $k$-simplex $\sigma_j = [v_0, \dots, v_k]$ is $\sum_{m=0}^{k} (-1)^m [v_0,
// /// \dots, \hat{v}_m, \dots, v_k]$. The coefficient for $\sigma_i^{\prime}$ (the $i$-th
// /// $(k-1)$-simplex) in $\partial \sigma_j$ is determined accordingly.
// ///
// /// # Arguments
// /// * `ordered_k_simplices`: A slice of [`Simplex`] objects, forming an ordered basis for the
// ///   $k$-chain group $C_k$.
// /// * `ordered_km1_simplices`: A slice of [`Simplex`] objects, forming an ordered basis for the
// ///   $(k-1)$-chain group $C_{k-1}$.
// ///
// /// # Type Parameters
// /// * `F`: The coefficient field, must implement [`Field`] and `Copy`.
// ///
// /// # Returns
// /// A `Vec<Vec<F>>` representing the boundary matrix. The matrix will have
// /// `ordered_km1_simplices.len()` rows and `ordered_k_simplices.len()` columns.
// /// Returns an empty or specially-dimensioned matrix if either basis is empty.
// pub fn get_boundary_matrix<F: Field + Copy>(
//   ordered_k_simplices: &[Simplex],   // Basis for C_k (domain)
//   ordered_km1_simplices: &[Simplex], // Basis for C_{k-1} (codomain)
// ) -> DynamicDenseMatrix<F, RowMajor> {
//   if ordered_k_simplices.is_empty() || ordered_km1_simplices.is_empty() {
//     return DynamicDenseMatrix::<F, RowMajor>::new();
//   }

//   let num_rows = ordered_km1_simplices.len();

//   let mut matrix = DynamicDenseMatrix::<F, RowMajor>::new();

//   // Create a map for quick lookup of (k-1)-simplex indices
//   let km1_simplex_to_idx: HashMap<Simplex, usize> =
//     ordered_km1_simplices.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

//   for k_simplex in ordered_k_simplices {
//     // For each k-simplex, compute its boundary.
//     // The boundary is a (k-1)-chain.
//     let boundary_chain =
//       Chain::from_simplex_and_coeff(k_simplex.clone(), <F as One>::one()).boundary();

//     // Convert this boundary_chain into a column vector for the matrix.
//     let col_vector = chain_to_coeff_vector(&boundary_chain, &km1_simplex_to_idx, num_rows);
//     matrix.append_column(&col_vector);
//   }
//   matrix
// }
