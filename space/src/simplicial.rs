//! A module for working with simplicial complexes and computing their boundaries.
//!
//! This module provides data structures and algorithms for working with simplicial complexes,
//! including operations for computing boundaries and manipulating chains of simplices.
//! The implementation ensures that vertices in simplices are always stored in sorted order.

use std::{
  collections::HashMap,
  ops::{Add, Neg},
};

use harness_algebra::ring::Field;
use itertools::Itertools;
use num_traits::{One, Zero};

/// A simplex represents a k-dimensional geometric object that is the convex hull of k+1 vertices.
///
/// For example:
/// - A 0-simplex is a point
/// - A 1-simplex is a line segment
/// - A 2-simplex is a triangle
/// - A 3-simplex is a tetrahedron
///
/// # Fields
/// * `vertices` - A sorted vector of vertex indices that define the simplex
/// * `dimension` - The dimension of the simplex (number of vertices - 1)
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Simplex {
  vertices:  Vec<usize>,
  dimension: usize,
}

impl Eq for Simplex {}

impl PartialOrd for Simplex {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Simplex {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.vertices.cmp(&other.vertices) }
}

impl Simplex {
  /// Creates a new simplex from k+1 vertices where k is the dimension.
  ///
  /// # Arguments
  /// * `dimension` - The dimension of the simplex
  /// * `vertices` - A vector of vertex indices
  ///
  /// # Panics
  /// * If the number of vertices does not equal dimension + 1
  /// * If any vertex indices are repeated
  pub fn new(dimension: usize, vertices: Vec<usize>) -> Self {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices: vertices.into_iter().sorted().collect(), dimension }
  }

  /// Returns a reference to the sorted vertices of the simplex.
  pub fn vertices(&self) -> &[usize] { &self.vertices }

  /// Returns the dimension of the simplex.
  pub fn dimension(&self) -> usize { self.dimension }

  /// Computes all (dimension-1)-dimensional faces of this simplex.
  ///
  /// For example, a triangle's faces are its three edges.
  pub fn faces(&self) -> Vec<Simplex> {
    self
      .vertices
      .clone()
      .into_iter()
      .combinations(self.dimension)
      .map(|v| Self::new(self.dimension - 1, v))
      .collect()
  }
}

/// A simplicial complex represents a collection of simplices that are properly glued together.
///
/// The complex stores simplices grouped by dimension, where each dimension's simplices
/// are stored in a vector at the corresponding index.
#[derive(Debug, Default)]
pub struct SimplicialComplex {
  /// Vector of simplices grouped by dimension
  simplices: HashMap<usize, Vec<Simplex>>,
}

impl SimplicialComplex {
  /// Creates a new empty simplicial complex.
  pub fn new() -> Self { Self { simplices: HashMap::new() } }

  /// Adds a simplex and all its faces to the complex.
  ///
  /// If the simplex is already present, it will not be added again.
  /// This method recursively adds all faces of the simplex as well.
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

  /// Computes the boundary of all simplices of a given dimension in the complex.
  ///
  /// # Type Parameters
  /// * `R` - The coefficient ring type (must implement Clone, Neg, Add, Zero, and One)
  ///
  /// # Arguments
  /// * `dimension` - The dimension of simplices whose boundary to compute
  ///
  /// # Returns
  /// A chain representing the boundary. If dimension is 0 or no simplices of that
  /// dimension exist in the complex, returns an empty chain.
  ///
  /// # Note
  /// This implementation's logic for choosing coefficients is specific and may not align with
  /// standard definitions for all purposes. It also assumes at most two simplices share any given
  /// face. The primary boundary operator for homology calculations is `Chain::boundary()`.
  pub fn boundary<R: Clone + Neg<Output = R> + Add<Output = R> + Zero + One>(
    &self,
    dimension: usize,
  ) -> Chain<R> {
    if dimension == 0 {
      return Chain::new();
    }
    let mut chain = Chain::new();
    if let Some(simplices_in_dim) = self.simplices.get(&dimension) {
      for s_val in simplices_in_dim.clone() {
        // Renamed simplex to s_val
        if chain.simplices.iter().flat_map(Simplex::faces).any(|f| s_val.faces().contains(&f)) {
          chain = chain + Chain::from_simplex_and_coeff(s_val, -R::one());
        } else {
          chain = chain + Chain::from_simplex_and_coeff(s_val, R::one());
        }
      }
    }
    chain.boundary()
  }

  /// Returns a reference to the simplices of a given dimension.
  ///
  /// # Arguments
  /// * `dimension` - The dimension of the simplices to return
  ///
  /// # Returns
  /// An `Option` containing a slice of simplices if the dimension exists,
  /// otherwise `None`.
  pub fn simplices_by_dimension(&self, dimension: usize) -> Option<&[Simplex]> {
    self.simplices.get(&dimension).map(Vec::as_slice)
  }

  /// Computes the k-th homology group H_k with Z2 coefficients.
  pub fn compute_homology<F: Field + Copy>(&self, k: usize) -> HomologyGroup<F> {
    if k == 0 {
      let mut vertices = match self.simplices_by_dimension(0) {
        Some(v) => v.to_vec(),
        None => return HomologyGroup::trivial(0),
      };
      if vertices.is_empty() {
        return HomologyGroup::trivial(0);
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
          h0_generators.push(Chain::from_simplex_and_coeff(vertices[i].clone(), <F as One>::one()));
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
      let z0_gens: Vec<Chain<F>> = vertices
        .iter()
        .map(|v| Chain::from_simplex_and_coeff(v.clone(), <F as One>::one()))
        .collect();
      let s1_for_b0 = match self.simplices_by_dimension(1) {
        Some(s) => {
          let mut sorted_s = s.to_vec();
          sorted_s.sort_unstable();
          sorted_s
        },
        None => Vec::new(),
      };
      let b0_gens = if s1_for_b0.is_empty() || vertices.is_empty() {
        Vec::new()
      } else {
        let mut mat_d1 = linalg::get_boundary_matrix(&s1_for_b0, &vertices);
        let (_, p_cols_d1) = linalg::row_gaussian_elimination(&mut mat_d1);
        let d_s1_chains: Vec<Chain<F>> = s1_for_b0
          .iter()
          .map(|s| Chain::from_simplex_and_coeff(s.clone(), <F as One>::one()).boundary())
          .collect();
        linalg::image_basis_from_row_echelon(&mat_d1, &d_s1_chains, &p_cols_d1)
      };
      return HomologyGroup {
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
      _ => return HomologyGroup::trivial(k),
    };
    let s_km1 = match self.simplices_by_dimension(k - 1) {
      Some(s) => {
        let mut ss = s.to_vec();
        ss.sort_unstable();
        ss
      },
      None => Vec::new(),
    };
    let s_kp1 = match self.simplices_by_dimension(k + 1) {
      Some(s) => {
        let mut ss = s.to_vec();
        ss.sort_unstable();
        ss
      },
      None => Vec::new(),
    };
    let ck_basis: Vec<Chain<F>> =
      s_k.iter().map(|s| Chain::from_simplex_and_coeff(s.clone(), <F as One>::one())).collect();

    let b_k_gens = if s_kp1.is_empty() || s_k.is_empty() {
      Vec::new()
    } else {
      let mut mat_dkp1 = linalg::get_boundary_matrix(&s_kp1, &s_k);
      let (_, p_cols) = linalg::row_gaussian_elimination(&mut mat_dkp1);
      let d_skp1_chains: Vec<Chain<F>> = s_kp1
        .iter()
        .map(|s| Chain::from_simplex_and_coeff(s.clone(), <F as One>::one()).boundary())
        .collect();
      linalg::image_basis_from_row_echelon(&mat_dkp1, &d_skp1_chains, &p_cols)
    };

    let z_k_gens = if s_k.is_empty() {
      Vec::new()
    } else if s_km1.is_empty() {
      ck_basis.clone()
    } else {
      let mut mat_dk = linalg::get_boundary_matrix(&s_k, &s_km1);
      let (_, p_cols) = linalg::row_gaussian_elimination(&mut mat_dk);
      linalg::kernel_basis_from_row_echelon(&mat_dk, &ck_basis, &p_cols)
    };

    if s_k.is_empty() {
      return HomologyGroup::trivial(k);
    }
    let sk_map: std::collections::HashMap<Simplex, usize> =
      s_k.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    let mut quot_mat_cols = Vec::new();
    for ch in b_k_gens.iter() {
      quot_mat_cols.push(linalg::chain_to_coeff_vector(ch, &sk_map, s_k.len()));
    }
    let num_b = quot_mat_cols.len();
    for ch in z_k_gens.iter() {
      quot_mat_cols.push(linalg::chain_to_coeff_vector(ch, &sk_map, s_k.len()));
    }

    if quot_mat_cols.is_empty() {
      return HomologyGroup {
        dimension:           k,
        betti_number:        z_k_gens.len(),
        cycle_generators:    z_k_gens.clone(),
        boundary_generators: b_k_gens,
        homology_generators: z_k_gens.clone(),
      };
    }

    let mut q_mat = vec![vec![<F as Zero>::zero(); quot_mat_cols.len()]; s_k.len()];
    if !s_k.is_empty() {
      for r in 0..s_k.len() {
        for c in 0..quot_mat_cols.len() {
          q_mat[r][c] = quot_mat_cols[c][r];
        }
      }
    }

    let (_, p_cols_q) = linalg::row_gaussian_elimination(&mut q_mat);
    let mut h_k_gens = Vec::new();
    for &p_idx in &p_cols_q {
      if p_idx >= num_b {
        h_k_gens.push(z_k_gens[p_idx - num_b].clone());
      }
    }

    HomologyGroup {
      dimension:           k,
      betti_number:        h_k_gens.len(),
      cycle_generators:    z_k_gens,
      boundary_generators: b_k_gens,
      homology_generators: h_k_gens,
    }
  }
}

/// A chain represents a formal sum of simplices with coefficients from a ring.
///
/// # Type Parameters
/// * `R` - The coefficient ring type
#[derive(Clone, Debug, Default)]
pub struct Chain<R> {
  /// The simplices in the chain
  simplices:    Vec<Simplex>,
  /// The coefficients corresponding to each simplex
  coefficients: Vec<R>,
}

impl<R> Chain<R> {
  /// Creates a new empty chain.
  pub fn new() -> Self { Self { simplices: vec![], coefficients: vec![] } }

  /// Creates a new chain with a single simplex and coefficient.
  pub fn from_simplex_and_coeff(simplex: Simplex, coeff: R) -> Self {
    Self { simplices: vec![simplex], coefficients: vec![coeff] }
  }

  /// Computes the boundary of this chain.
  ///
  /// The boundary operator satisfies the property that ∂² = 0,
  /// meaning the boundary of a boundary is empty.
  pub fn boundary(&self) -> Self
  where R: Clone + Neg<Output = R> + Add<Output = R> + Zero {
    let mut boundary = Self::new();
    for (coeff, simplex) in self.coefficients.clone().into_iter().zip(self.simplices.iter()) {
      for i in 0..=simplex.dimension() {
        let mut vertices = simplex.vertices().to_vec();
        let _ = vertices.remove(i);
        let face = Simplex::new(simplex.dimension() - 1, vertices);
        let chain = Self::from_simplex_and_coeff(
          face,
          if i % 2 == 0 { coeff.clone() } else { -coeff.clone() },
        );
        boundary = boundary + chain;
      }
    }
    boundary
  }
}

impl<R: Clone + PartialEq> PartialEq for Chain<R> {
  /// Checks if two chains are equal.
  ///
  /// Two chains are equal if they have the same simplices with the same coefficients,
  /// taking into account the orientation of the simplices.
  fn eq(&self, other: &Self) -> bool {
    let self_chain = self.coefficients.clone().into_iter().zip(self.simplices.iter());
    let other_chain = other.coefficients.clone().into_iter().zip(other.simplices.iter());

    self_chain.zip(other_chain).all(|((coeff_a, simplex_a), (coeff_b, simplex_b))| {
      coeff_a == coeff_b
        && permutation_sign(simplex_a.vertices()) == permutation_sign(simplex_b.vertices())
        && simplex_a.vertices() == simplex_b.vertices()
    })
  }
}

impl<R: Add<Output = R> + Neg<Output = R> + Clone + Zero> Add for Chain<R> {
  type Output = Self;

  /// Adds two chains together.
  ///
  /// The addition combines like terms (simplices) by adding their coefficients.
  /// Terms with zero coefficients are removed from the result.
  ///
  /// # Note
  /// This implementation assumes the chains contain simplices of the same dimension.
  fn add(self, other: Self) -> Self::Output {
    let mut result_simplices = Vec::new();
    let mut result_coefficients: Vec<R> = Vec::new();

    // Add all simplices from self
    for (idx, simplex) in self.simplices.iter().enumerate() {
      let coefficient = self.coefficients[idx].clone();

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, res_simplex) in result_simplices.iter().enumerate() {
        if simplex == res_simplex {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx].clone() + coefficient.clone();
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_simplices.push(simplex.clone());
        result_coefficients.push(coefficient);
      }
    }

    // Add all simplices from other
    for (idx, simplex) in other.simplices.iter().enumerate() {
      let coefficient = other.coefficients[idx].clone();

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, res_simplex) in result_simplices.iter().enumerate() {
        if simplex == res_simplex {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx].clone() + coefficient.clone();
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_simplices.push(simplex.clone());
        result_coefficients.push(coefficient);
      }
    }

    // Filter out zero coefficients
    let mut i = 0;
    while i < result_coefficients.len() {
      if result_coefficients[i].is_zero() {
        // Remove zero coefficients and their corresponding simplices
        result_coefficients.remove(i);
        result_simplices.remove(i);
        // Don't increment i since we've shifted the array
      } else {
        i += 1;
      }
    }

    Self { simplices: result_simplices, coefficients: result_coefficients }
  }
}

/// Represents whether a permutation is odd or even.
#[derive(Debug, PartialEq, Eq)]
pub enum Permutation {
  /// An odd permutation
  Odd,
  /// An even permutation
  Even,
}

/// Computes the sign of a permutation by counting inversions.
///
/// # Arguments
/// * `item` - A slice of ordered items
///
/// # Returns
/// `Permutation::Even` if the number of inversions is even,
/// `Permutation::Odd` if the number of inversions is odd.
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

// TODO: This could be done over a ring in general, but a `Field` is much nicer.
/// Stores the results of a homology computation for a specific dimension.
///
/// # Type Parameters
/// * `R` - The coefficient ring type.
#[derive(Debug, Clone)]
pub struct HomologyGroup<R: Field> {
  /// The dimension k for which this homology group H_k is computed.
  pub dimension:           usize,
  /// The Betti number, which is the rank of the homology group H_k.
  /// For Z2 coefficients, this is dim(H_k).
  pub betti_number:        usize,
  /// A basis for the k-cycles (Z_k = ker ∂_k).
  /// Each element is a chain representing a cycle.
  pub cycle_generators:    Vec<Chain<R>>,
  /// A basis for the k-boundaries (B_k = im ∂_{k+1}).
  /// Each element is a chain representing a boundary.
  pub boundary_generators: Vec<Chain<R>>,
  /// A basis for the homology group H_k = Z_k / B_k.
  /// Each element is a chain representing a homology class generator.
  pub homology_generators: Vec<Chain<R>>,
}

impl<R: Field> HomologyGroup<R> {
  /// Creates a new, empty homology group for a given dimension.
  /// Typically used when the homology group is trivial.
  pub fn trivial(dimension: usize) -> Self {
    Self {
      dimension,
      betti_number: 0,
      cycle_generators: Vec::new(),
      boundary_generators: Vec::new(),
      homology_generators: Vec::new(),
    }
  }
}

mod linalg {
  use std::collections::HashMap;

  use num_traits::{One, Zero};

  use super::{Chain, Field, Simplex};

  /// Performs Gaussian elimination on a matrix over Z2 to bring it to column echelon form.
  /// The matrix is modified in place.
  /// Returns the rank of the matrix (number of pivot columns).
  pub fn column_gaussian_elimination<F: Field + Copy>(matrix: &mut Vec<Vec<F>>) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
      return 0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut pivot_row = 0;
    let mut rank = 0;

    for j in 0..cols {
      // Iterate through columns (potential pivot columns)
      if pivot_row >= rows {
        break;
      }
      let mut i = pivot_row;
      while i < rows && matrix[i][j].is_zero() {
        i += 1;
      }

      if i < rows {
        // Found a pivot in this column at matrix[i][j]
        // Swap row i with pivot_row to bring pivot to matrix[pivot_row][j]
        if i != pivot_row {
          for col_idx in j..cols {
            let temp = matrix[i][col_idx];
            matrix[i][col_idx] = matrix[pivot_row][col_idx];
            matrix[pivot_row][col_idx] = temp;
          }
        }

        // Eliminate other 1s in this column below the pivot
        // (Not strictly needed for column echelon form if we only clear to the right for kernel,
        // but for a canonical form or image basis, this structure is fine)
        // For Z2, if matrix[k][j] is 1 (and k != pivot_row), add pivot_row to row k.
        for k in 0..rows {
          if k != pivot_row && !matrix[k][j].is_zero() {
            for col_idx in j..cols {
              matrix[k][col_idx] = matrix[k][col_idx] + matrix[pivot_row][col_idx];
            }
          }
        }
        pivot_row += 1;
        rank += 1;
      }
    }
    rank
  }

  /// Performs Gaussian elimination on a matrix over Z2 to bring it to row echelon form.
  /// The matrix is modified in place.
  /// Returns a tuple (rank, pivot_columns_indices).
  pub fn row_gaussian_elimination<F: Field + Copy>(
    matrix: &mut Vec<Vec<F>>,
  ) -> (usize, Vec<usize>) {
    if matrix.is_empty() || matrix[0].is_empty() {
      return (0, Vec::new());
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut lead = 0; // current pivot column
    let mut rank = 0;
    let mut pivot_cols = Vec::new();

    for r in 0..rows {
      if lead >= cols {
        break;
      }
      let mut i = r;
      while matrix[i][lead].is_zero() {
        i += 1;
        if i == rows {
          i = r;
          lead += 1;
          if lead == cols {
            return (rank, pivot_cols);
          }
        }
      }
      matrix.swap(i, r);
      pivot_cols.push(lead);

      // Normalize pivot row (pivot element is already 1 in Z2 if non-zero)
      // For rows below pivot, eliminate the entry in the pivot column
      for i in 0..rows {
        if i != r && !matrix[i][lead].is_zero() {
          // matrix[i] = matrix[i] + matrix[r]
          for j in lead..cols {
            matrix[i][j] = matrix[i][j] + matrix[r][j];
          }
        }
      }
      lead += 1;
      rank += 1;
    }
    (rank, pivot_cols)
  }

  /// Extracts a basis for the kernel (null space) of a matrix A (m x n) given in row echelon form.
  /// `column_basis_elements` are chains corresponding to columns of original matrix A.
  /// The number of elements in `column_basis_elements` must be `n` (number of columns).
  pub fn kernel_basis_from_row_echelon<F: Field + Copy>(
    row_echelon_matrix: &[Vec<F>],
    column_basis_elements: &[Chain<F>],
    pivot_cols: &[usize],
  ) -> Vec<Chain<F>> {
    if row_echelon_matrix.is_empty() || row_echelon_matrix[0].is_empty() {
      // If matrix is empty, or has 0 columns, kernel depends on context.
      // If 0 columns, kernel is trivial. If 0 rows (non-empty cols), kernel is all of C_n.
      // This case needs careful handling based on matrix dimensions.
      // For now, if matrix is empty, assume trivial kernel if no column_basis_elements,
      // otherwise it's more complex. Let's assume non-empty matrix or column_basis_elements match
      // cols.
      if column_basis_elements.is_empty() {
        return Vec::new();
      }
      // If it's a zero map from non-trivial C_n, then all of C_n is kernel
      return column_basis_elements.to_vec();
    }

    let num_rows = row_echelon_matrix.len();
    let num_cols = row_echelon_matrix[0].len();
    assert_eq!(
      num_cols,
      column_basis_elements.len(),
      "Number of columns in matrix must match number of column basis elements"
    );

    let mut kernel_generators = Vec::new();
    let mut current_pivot_idx = 0;

    for j in 0..num_cols {
      // Iterate through columns
      if current_pivot_idx < pivot_cols.len() && pivot_cols[current_pivot_idx] == j {
        // This is a pivot column
        current_pivot_idx += 1;
      } else {
        // This is a free variable column
        let mut generator_coeffs = vec![<F as Zero>::zero(); num_cols];
        generator_coeffs[j] = <F as One>::one(); // Set free variable to 1

        // Solve for pivot variables in terms of this free variable
        // Ax = 0. For row i, Sum(A_ik * x_k) = 0
        // Iterate upwards through pivot rows to express pivot variables.
        let mut temp_pivot_idx = 0;
        for r in 0..num_rows {
          // iterate through effective rows (rank)
          if temp_pivot_idx < pivot_cols.len()
            && row_echelon_matrix[r][pivot_cols[temp_pivot_idx]].is_one()
          {
            let pivot_col_for_row_r = pivot_cols[temp_pivot_idx];
            // If this pivot var is not the free var itself (it shouldn't be)
            if pivot_col_for_row_r != j {
              // The value of x_{pivot_col_for_row_r} should be such that equation for row r holds.
              // A[r][pivot_col_for_row_r]*x_{pivot_col_for_row_r} + A[r][j]*x_j (which is 1) + ...
              // = 0 Since A[r][pivot_col_for_row_r] is 1 (pivot), and x_j is 1:
              // x_{pivot_col_for_row_r} + A[r][j] = 0  => x_{pivot_col_for_row_r} = -A[r][j] =
              // A[r][j] in Z2.
              if !row_echelon_matrix[r][j].is_zero() {
                // only if A[r][j] is 1
                generator_coeffs[pivot_col_for_row_r] = row_echelon_matrix[r][j];
              }
            }
            temp_pivot_idx += 1;
            if temp_pivot_idx >= pivot_cols.len() {
              break;
            }
          } else if temp_pivot_idx >= pivot_cols.len() {
            break; // No more pivots
          }
        }

        // Construct chain from coefficients
        let mut gen_chain = Chain::new();
        for (idx, coeff) in generator_coeffs.iter().enumerate() {
          if !coeff.is_zero() {
            // Create a unit chain for the simplex column_basis_elements[idx]
            // This assumes column_basis_elements[idx] is a Chain with one simplex and coeff 1.
            // Or, more generally, it is the chain we want to add if coeff is 1.
            // For now, let's assume column_basis_elements are unit chains.
            gen_chain = gen_chain + column_basis_elements[idx].clone(); // Coeff is Z2(1), so adding
                                                                        // is correct.
          }
        }
        if !gen_chain.simplices.is_empty() {
          // Ensure it's not a zero chain
          kernel_generators.push(gen_chain);
        }
      }
    }
    kernel_generators
  }

  /// Extracts a basis for the image (column space) of a matrix A.
  /// The columns of the original matrix A that become pivot columns in its Row Echelon Form
  /// correspond to basis vectors for Im(A). `column_basis_elements` are chains corresponding to
  /// columns of original matrix A.
  pub fn image_basis_from_row_echelon<F: Field + Copy>(
    _row_echelon_matrix: &[Vec<F>], // Matrix itself not strictly needed if we have pivot_cols
    column_basis_elements: &[Chain<F>],
    pivot_cols: &[usize],
  ) -> Vec<Chain<F>> {
    let mut image_generators = Vec::new();
    for &pivot_col_idx in pivot_cols {
      if pivot_col_idx < column_basis_elements.len() {
        image_generators.push(column_basis_elements[pivot_col_idx].clone());
      }
    }
    image_generators
  }

  // Helper function to convert a Chain to a coefficient vector relative to a basis of simplices.
  // Used internally for matrix construction.
  pub(super) fn chain_to_coeff_vector<F: Field + Copy>(
    chain: &Chain<F>,
    basis_simplices_map: &HashMap<Simplex, usize>, // Map simplex to its index in the basis
    basis_size: usize,
  ) -> Vec<F> {
    let mut vector = vec![<F as Zero>::zero(); basis_size];
    for (i, simplex) in chain.simplices.iter().enumerate() {
      if let Some(&idx) = basis_simplices_map.get(simplex) {
        // In Z2, the coefficient is either 0 or 1.
        // If the simplex is in the chain, its coefficient is effectively Z2(1)
        // (assuming chain.coefficients[i] is Z2::one()).
        // Chain addition takes care of summing coefficients.
        if !chain.coefficients[i].is_zero() {
          vector[idx] = vector[idx] + chain.coefficients[i]; // Add in Z2
        }
      }
    }
    vector
  }

  /// Constructs the boundary matrix for ∂_k: C_k -> C_{k-1}.
  /// Columns are indexed by `ordered_k_simplices`.
  /// Rows are indexed by `ordered_km1_simplices`.
  /// The (i,j)-th entry is 1 if ordered_km1_simplices[i] is in the boundary of
  /// ordered_k_simplices[j], else 0.
  pub fn get_boundary_matrix<F: Field + Copy>(
    ordered_k_simplices: &[Simplex],   // Basis for C_k (domain)
    ordered_km1_simplices: &[Simplex], // Basis for C_{k-1} (codomain)
  ) -> Vec<Vec<F>> {
    if ordered_k_simplices.is_empty() {
      // No k-simplices, so map is from a zero space. Matrix has 0 columns.
      // Number of rows is |S_{k-1}|.
      return vec![vec![]; ordered_km1_simplices.len()];
    }
    if ordered_km1_simplices.is_empty() {
      // Map is to a zero space (C_{k-1} is trivial).
      // Matrix has |S_k| columns and 0 rows.
      // Each column is empty.
      // However, conventionally, this matrix would have 0 rows and |S_k| columns.
      return vec![Vec::new(); 0]; // Standard representation of a matrix with 0 rows.
    }

    let num_rows = ordered_km1_simplices.len();
    let num_cols = ordered_k_simplices.len();

    let mut matrix = vec![vec![<F as Zero>::zero(); num_cols]; num_rows];

    // Create a map for quick lookup of (k-1)-simplex indices
    let km1_simplex_to_idx: HashMap<Simplex, usize> =
      ordered_km1_simplices.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    for (j, k_simplex) in ordered_k_simplices.iter().enumerate() {
      // For each k-simplex, compute its boundary.
      // The boundary is a (k-1)-chain.
      let boundary_chain =
        Chain::from_simplex_and_coeff(k_simplex.clone(), <F as One>::one()).boundary();

      // Convert this boundary_chain into a column vector for the matrix.
      let col_vector = chain_to_coeff_vector(&boundary_chain, &km1_simplex_to_idx, num_rows);
      for i in 0..num_rows {
        matrix[i][j] = col_vector[i];
      }
    }
    matrix
  }
}

#[cfg(test)]
mod tests {
  use harness_algebra::arithmetic::Boolean;

  use super::*;

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

    let chain1 = Chain::from_simplex_and_coeff(simplex1, 1);
    let chain2 = Chain::from_simplex_and_coeff(simplex2, 2);

    let result = chain1 + chain2;

    assert_eq!(result.simplices.len(), 2);
    assert_eq!(result.coefficients.len(), 2);
    assert_eq!(result.simplices[0].vertices(), &[0, 1]);
    assert_eq!(result.simplices[1].vertices(), &[1, 2]);
    assert_eq!(result.coefficients[0], 1);
    assert_eq!(result.coefficients[1], 2);
  }

  #[test]
  fn test_chain_addition_same_simplex() {
    // Create two chains with the same simplex
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);

    let chain1 = Chain::from_simplex_and_coeff(simplex1, 2);
    let chain2 = Chain::from_simplex_and_coeff(simplex2, 3);

    let result = chain1 + chain2;

    assert_eq!(result.simplices.len(), 1);
    assert_eq!(result.coefficients.len(), 1);
    assert_eq!(result.simplices[0].vertices(), &[0, 1]);
    assert_eq!(result.coefficients[0], 5); // 2 + 3 = 5
  }

  #[test]
  fn test_chain_addition_canceling_coefficients() {
    // Create two chains with the same simplex but opposite coefficients
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]);

    let chain1 = Chain::from_simplex_and_coeff(simplex1, 2);
    let chain2 = Chain::from_simplex_and_coeff(simplex2, -2);

    let result = chain1 + chain2;

    // The result should be empty since the coefficients cancel out
    assert_eq!(result.simplices.len(), 0);
    assert_eq!(result.coefficients.len(), 0);
  }

  #[test]
  fn test_chain_boundary_edge() {
    // The boundary of an edge is its two vertices with opposite signs
    let edge = Simplex::new(1, vec![0, 1]);
    let chain = Chain::from_simplex_and_coeff(edge, 1);

    let boundary = chain.boundary();

    // Should have two 0-simplices (vertices)
    assert_eq!(boundary.simplices.len(), 2);
    assert_eq!(boundary.coefficients.len(), 2);

    // Verify the vertices
    assert!(boundary.simplices.iter().any(|s| s.vertices().contains(&0)));
    assert!(boundary.simplices.iter().any(|s| s.vertices().contains(&1)));
    assert!(boundary.simplices.len() == 2);

    // Verify opposite signs (exact sign depends on your implementation)
    assert_eq!(boundary.coefficients[0] + boundary.coefficients[1], 0);
  }

  #[test]
  fn test_chain_boundary_triangle() {
    // The boundary of a triangle is its three edges
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let chain = Chain::from_simplex_and_coeff(triangle, 1);

    let boundary = chain.boundary();

    // Should have three 1-simplices (edges)
    assert_eq!(boundary.simplices.len(), 3);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.simplices.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_boundary_squared_is_zero() {
    // Verify that ∂² = 0 for a triangle
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let chain = Chain::from_simplex_and_coeff(triangle, 1);

    let boundary = chain.boundary();
    let boundary_squared = boundary.boundary();

    // Boundary of boundary should be empty (∂² = 0)
    assert_eq!(boundary_squared.simplices.len(), 0);
    assert_eq!(boundary_squared.coefficients.len(), 0);
  }

  #[test]
  fn test_complex_chain_operations() {
    // Create a 2-chain with two triangles sharing an edge
    let triangle1 = Simplex::new(2, vec![0, 1, 2]);
    let triangle2 = Simplex::new(2, vec![1, 2, 3]);

    let chain1 = Chain::from_simplex_and_coeff(triangle1, 1);
    let chain2 = Chain::from_simplex_and_coeff(triangle2, -1);

    let combined_chain = chain1 + chain2;
    let boundary = combined_chain.boundary();

    // The boundary should have 4 edges (the shared edge [1,2] cancels out)
    assert_eq!(boundary.simplices.len(), 4);

    // Verify the edges
    let edge_vertices: Vec<Vec<usize>> =
      boundary.simplices.iter().map(|s| s.vertices().to_vec()).collect();

    assert!(edge_vertices.contains(&vec![0, 1]));
    assert!(edge_vertices.contains(&vec![0, 2]));
    assert!(edge_vertices.contains(&vec![1, 3]));
    assert!(edge_vertices.contains(&vec![2, 3]));

    // The shared edge [1,2] should not be present because its coefficients cancel
    assert!(!edge_vertices.contains(&vec![1, 2]));
  }

  #[test]
  fn test_simplicial_complex_boundary() {
    let mut complex = SimplicialComplex::new();
    complex.join_simplex(Simplex::new(2, vec![0, 1, 2]));
    complex.join_simplex(Simplex::new(2, vec![1, 2, 3]));
    let boundary: Chain<i32> = complex.boundary(2);
    assert_eq!(boundary.simplices.len(), 4);
    assert_eq!(boundary.coefficients.len(), 4);
    assert_eq!(boundary.simplices[0].vertices(), &[0, 2]);
    assert_eq!(boundary.simplices[1].vertices(), &[0, 1]);
    assert_eq!(boundary.simplices[2].vertices(), &[2, 3]);
    assert_eq!(boundary.simplices[3].vertices(), &[1, 3]);
    assert_eq!(boundary.coefficients[0], -1);
    assert_eq!(boundary.coefficients[1], 1);
    assert_eq!(boundary.coefficients[2], -1);
    assert_eq!(boundary.coefficients[3], 1);
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

  #[test]
  fn test_homology_point() {
    let mut complex = SimplicialComplex::new();
    let p0 = Simplex::new(0, vec![0]);
    complex.join_simplex(p0.clone());

    // H_0
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");
    // The generator should be the point [0]
    let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, Boolean::one());
    assert!(h0.homology_generators.contains(&expected_gen_h0), "H0: Generator mismatch");

    // H_1
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
    assert!(h1.homology_generators.is_empty(), "H1: Should have no generators");

    // H_2
    let h2 = complex.compute_homology::<Boolean>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti number for a point should be 0");
  }

  #[test]
  fn test_homology_edge() {
    let mut complex = SimplicialComplex::new();
    let edge01 = Simplex::new(1, vec![0, 1]);
    complex.join_simplex(edge01.clone()); // join_simplex adds faces: p0, p1

    // H_0
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for an edge should be 1 (connected)");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");
    // Generator could be [0] or [1]. Let's check if it's one of them.
    // The compute_homology_z2(0) implementation picks the first vertex of a component.
    // Since vertices are sorted (0, then 1), [0] should be the generator.
    let p0 = Simplex::new(0, vec![0]);
    let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, Boolean::one());
    assert!(h0.homology_generators.contains(&expected_gen_h0), "H0: Generator for edge");

    // H_1
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for an edge should be 0 (no hole)");
    assert!(h1.homology_generators.is_empty(), "H1: Should have no generators for an edge");
  }

  #[test]
  fn test_homology_two_disjoint_points() {
    let mut complex = SimplicialComplex::new();
    let p0 = Simplex::new(0, vec![0]);
    let p1 = Simplex::new(0, vec![1]);
    complex.join_simplex(p0.clone());
    complex.join_simplex(p1.clone());

    // H_0
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 2, "H0: Betti for two disjoint points should be 2");
    assert_eq!(h0.homology_generators.len(), 2, "H0: Should have two generators");
    let expected_gen1_h0 = Chain::from_simplex_and_coeff(p0, Boolean::one());
    let expected_gen2_h0 = Chain::from_simplex_and_coeff(p1, Boolean::one());
    assert!(h0.homology_generators.contains(&expected_gen1_h0), "H0: Generator [0] missing");
    assert!(h0.homology_generators.contains(&expected_gen2_h0), "H0: Generator [1] missing");

    // H_1
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti for two disjoint points should be 0");
  }

  #[test]
  fn test_homology_filled_triangle() {
    let mut complex = SimplicialComplex::new();
    let triangle012 = Simplex::new(2, vec![0, 1, 2]);
    complex.join_simplex(triangle012.clone()); // Adds tri, its edges, and its vertices

    // H_0
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for a triangle should be 1");

    // H_1
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti for a filled triangle should be 0 (no 1D hole)");
    assert!(h1.homology_generators.is_empty(), "H1: Should have no 1D homology generators");

    // H_2
    let h2 = complex.compute_homology::<Boolean>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for a filled triangle should be 0 (not a void)");
    assert!(h2.homology_generators.is_empty(), "H2: Should have no 2D homology generators");
  }

  #[test]
  fn test_homology_circle() {
    let mut complex = SimplicialComplex::new();
    // A circle formed by three 1-simplices (edges of a triangle)
    let s01 = Simplex::new(1, vec![0, 1]);
    let s12 = Simplex::new(1, vec![1, 2]);
    let s20 = Simplex::new(1, vec![0, 2]); // Note: vertices sorted to [0,2]

    complex.join_simplex(s01.clone());
    complex.join_simplex(s12.clone());
    complex.join_simplex(s20.clone());
    // This also adds vertices [0], [1], [2]

    // H_0
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for a circle should be 1");

    // H_1
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.betti_number, 1, "H1: Betti for a circle should be 1");
    assert_eq!(h1.homology_generators.len(), 1, "H1: Should have one 1D homology generator");

    // Check the generator for H1. It should be the cycle s01+s12+s02 (in Z2).
    // The generator chain must contain these three simplices s01, s12, s02 with Z2(1) coeff.
    // And its boundary must be zero.
    let generator_h1 = h1.homology_generators[0].clone();
    assert_eq!(generator_h1.simplices.len(), 3, "H1: Circle generator should have 3 simplices");

    // Define the simplices we expect to find in the generator chain
    let simplex01 = Simplex::new(1, vec![0, 1]);
    let simplex12 = Simplex::new(1, vec![1, 2]);
    let simplex02 = Simplex::new(1, vec![0, 2]);

    assert!(generator_h1.simplices.contains(&simplex01), "H1: Gen missing s01 ([0,1])");
    assert!(generator_h1.simplices.contains(&simplex12), "H1: Gen missing s12 ([1,2])");
    assert!(generator_h1.simplices.contains(&simplex02), "H1: Gen missing s02 ([0,2])");

    for coeff in &generator_h1.coefficients {
      assert_eq!(*coeff, Boolean::one(), "H1: All coeffs in Z2 cycle generator should be 1");
    }
    assert!(
      generator_h1.boundary().simplices.is_empty(),
      "H1: Generator's boundary should be zero"
    );

    // H_2
    let h2 = complex.compute_homology::<Boolean>(2);
    assert_eq!(h2.betti_number, 0, "H2: Betti for a circle should be 0");
  }

  #[test]
  fn test_homology_sphere_surface() {
    // Boundary of a tetrahedron: 4 vertices, 6 edges, 4 faces ([012],[013],[023],[123])
    let mut complex = SimplicialComplex::new();
    let f012 = Simplex::new(2, vec![0, 1, 2]);
    let f013 = Simplex::new(2, vec![0, 1, 3]);
    let f023 = Simplex::new(2, vec![0, 2, 3]);
    let f123 = Simplex::new(2, vec![1, 2, 3]);

    // Important: Only add the 2-simplices (faces).
    // join_simplex will add their boundaries (edges and vertices).
    // If we added the 3-simplex (tetrahedron itself), H2 would be 0.
    complex.join_simplex(f012.clone());
    complex.join_simplex(f013.clone());
    complex.join_simplex(f023.clone());
    complex.join_simplex(f123.clone());

    // H_0: Should be 1 (connected surface)
    let h0 = complex.compute_homology::<Boolean>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for sphere surface should be 1");

    // H_1: Should be 0 (no 1D holes on sphere surface)
    let h1 = complex.compute_homology::<Boolean>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti for sphere surface should be 0");
    assert!(h1.homology_generators.is_empty(), "H1: Generators for sphere surface should be empty");

    // H_2: Should be 1 (the enclosed void)
    let h2 = complex.compute_homology::<Boolean>(2);
    assert_eq!(h2.betti_number, 1, "H2: Betti for sphere surface should be 1");
    assert_eq!(h2.homology_generators.len(), 1, "H2: Should have one 2D generator");

    // The generator should be the sum of all 4 faces (with Z2 coefficients)
    let generator_h2 = h2.homology_generators[0].clone();
    assert_eq!(generator_h2.simplices.len(), 4, "H2: Sphere generator should be sum of 4 faces");
    assert!(generator_h2.simplices.contains(&f012), "H2: Gen missing face [0,1,2]");
    assert!(generator_h2.simplices.contains(&f013), "H2: Gen missing face [0,1,3]");
    assert!(generator_h2.simplices.contains(&f023), "H2: Gen missing face [0,2,3]");
    assert!(generator_h2.simplices.contains(&f123), "H2: Gen missing face [1,2,3]");
    for coeff in &generator_h2.coefficients {
      assert_eq!(*coeff, Boolean::one(), "H2: All coeffs in Z2 cycle generator should be 1");
    }
    // Boundary of this 2-cycle must be 0.
    assert!(
      generator_h2.boundary().simplices.is_empty(),
      "H2: Generator's boundary should be zero"
    );

    // H_3: No 3-simplices in the complex, so H_3 is trivial.
    let h3 = complex.compute_homology::<Boolean>(3);
    assert_eq!(h3.betti_number, 0, "H3: Betti for sphere surface (dim 2 complex) should be 0");
  }
}
