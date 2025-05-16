//! A module for working with simplicial complexes and computing their boundaries.
//!
//! This module provides data structures and algorithms for working with simplicial complexes,
//! including operations for computing boundaries and manipulating chains of simplices.
//! The implementation ensures that vertices in simplices are always stored in sorted order.

use std::{
  collections::HashMap,
  ops::{Add, Neg},
};

use harness_algebra::{arithmetic::Boolean, modular, prime_field, ring::Field};
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

      let pivot_val = matrix[r][lead];
      // For a field, a non-zero pivot_val is expected here.
      // The Field trait should provide inverse. Panicking if not found for a non-zero element is
      // acceptable.
      let inv_pivot = pivot_val.multiplicative_inverse();

      // Normalize pivot row: matrix[r][j] = matrix[r][j] * inv_pivot
      for j in lead..cols {
        matrix[r][j] = matrix[r][j] * inv_pivot;
      }

      // Eliminate other rows: matrix[i] = matrix[i] - factor * matrix[r]
      for i_row in 0..rows {
        if i_row != r {
          let factor = matrix[i_row][lead]; // factor is F, which is Copy
          if !factor.is_zero() {
            for j_col in lead..cols {
              let term = factor * matrix[r][j_col];
              matrix[i_row][j_col] = matrix[i_row][j_col] - term;
            }
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
      if column_basis_elements.is_empty() {
        return Vec::new();
      }
      // If it's a zero map from non-trivial C_n to C_m (m > 0), or to C_0,
      // then all of C_n is kernel.
      // This case should be handled if num_rows (effective rank) is 0 but num_cols > 0.
      // The current logic for free variables should correctly make all columns free if rank is 0.
      let is_zero_matrix_effectively = pivot_cols.is_empty();
      if is_zero_matrix_effectively && !column_basis_elements.is_empty() {
        return column_basis_elements.to_vec();
      }
      if column_basis_elements.is_empty() {
        // Explicitly handle if no columns to form basis from
        return Vec::new();
      }
      // Fallback if matrix had rows but all were zero, or other edge cases
      // This part of the original condition might be too simple.
      // Let the main loop handle it: if all columns are free, it will generate all basis elements.
    }

    let num_cols = if !row_echelon_matrix.is_empty() { row_echelon_matrix[0].len() } else { 0 };
    if num_cols == 0 && column_basis_elements.is_empty() {
      return Vec::new();
    }
    // If num_cols is 0 but column_basis_elements is not, it's an inconsistency.
    // If row_echelon_matrix is empty but column_basis_elements is not, it implies a map to C_0.
    // In this case, all columns are free. num_rows would be 0.

    assert_eq!(
      num_cols,
      column_basis_elements.len(),
      "Number of columns in matrix must match number of column basis elements"
    );

    let mut kernel_generators = Vec::new();
    let mut current_pivot_idx = 0;

    for j_col in 0..num_cols {
      // Iterate through columns (potential free variables)
      if current_pivot_idx < pivot_cols.len() && pivot_cols[current_pivot_idx] == j_col {
        // This is a pivot column
        current_pivot_idx += 1;
      } else {
        // This is a free variable column j_col
        let mut generator_coeffs = vec![<F as Zero>::zero(); num_cols];
        generator_coeffs[j_col] = <F as One>::one(); // Set free variable x_{j_col} to 1

        // Solve for pivot variables in terms of this free variable
        // For each pivot row r, associated with a pivot column p_c = pivot_cols[r_idx]:
        //   x_{p_c} * 1 + M[r][j_col] * x_{j_col} = 0  (other free vars are 0, other pivots are 0
        // in this equation)   x_{p_c} = -M[r][j_col] * x_{j_col}
        // Since x_{j_col} = 1, then x_{p_c} = -M[r][j_col]

        // Iterate through the pivot rows of the row_echelon_matrix
        // The number of actual pivot rows is pivot_cols.len() (which is the rank)
        // `row_echelon_matrix` still has all original rows, but non-pivot rows are zero or become
        // zero relevant to pivots.

        let mut temp_p_idx = 0; // index into pivot_cols vector
        for r_idx in 0..pivot_cols.len() {
          // Iterate over actual pivot rows by their index in pivot_cols
          let pivot_row_actual_index = r_idx; // In reduced row echelon form, pivot r is in row r.
                                              // The way `row_gaussian_elimination` is written, it processes row by row.
                                              // The pivot `matrix[r][lead]` means `pivot_cols[r_idx]` is that `lead`.
          let pivot_col_for_this_row = pivot_cols[r_idx];

          // We need to ensure row_echelon_matrix[pivot_row_actual_index] is the correct row.
          // The RRE matrix has its pivots matrix[r_idx][pivot_cols[r_idx]] = 1.
          if !row_echelon_matrix[pivot_row_actual_index][j_col].is_zero() {
            generator_coeffs[pivot_col_for_this_row] =
              -row_echelon_matrix[pivot_row_actual_index][j_col];
          }
        }

        let mut gen_chain = Chain::new();
        for (idx, coeff) in generator_coeffs.iter().enumerate() {
          if !coeff.is_zero() {
            // This creates a chain with one simplex and the calculated coefficient.
            // Then adds it to gen_chain. This is not quite right.
            // We need to sum up (coeff_i * basis_chain_i).
            // gen_chain = gen_chain + (coeff.clone() * column_basis_elements[idx].clone());
            // Chain does not currently support multiplication by scalar from the left.
            // For now, assume coeff is 1 or -1, or for Z2, just 1.
            // If coeff is F::one(), add column_basis_elements[idx]
            // If coeff is -F::one(), subtract column_basis_elements[idx] (add its negation)
            // This logic needs Chain to support scalar multiplication or careful construction.

            // Current Chain::add combines like terms. We are building a new chain from a linear
            // combination. Let's assume column_basis_elements are single simplices with
            // coeff 1.
            let mut single_term_chain = column_basis_elements[idx].clone();
            // We need to scale single_term_chain by *coeff.
            // A simple way for now, if we assume Field allows it:
            for c in single_term_chain.coefficients.iter_mut() {
              *c = *c * (*coeff); // Assumes coeff is F, c is F.
            }
            if !coeff.is_zero() {
              // Re-check after potential multiplication if coeff was complex.
              gen_chain = gen_chain + single_term_chain;
            }
          }
        }
        if !gen_chain.simplices.is_empty() {
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

// Define Mod7 field
modular!(Mod7, u32, 7);
prime_field!(Mod7);

// Ensure Mod7 implements necessary traits if not covered by macros,
// e.g., Copy, Debug, PartialEq, Default. Field usually implies many.
// prime_field! likely derives these or ensures they are implemented.

#[cfg(test)]
mod tests {
  use std::fmt::Debug; // For F in assert messages if Chain prints it

  use harness_algebra::arithmetic::Boolean;
  use harness_algebra::ring::Field; // For trait bounds
  use num_traits::{One, Zero}; // For F::one(), F::zero()

  use super::{Mod7, *}; // Make sure Mod7 is in scope here

  // Helper trait bound alias for tests
  trait TestField: Field + Copy + PartialEq + Debug {}
  impl<T: Field + Copy + PartialEq + Debug> TestField for T {}

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

  fn test_homology_point_generic<F: TestField>() {
    let mut complex = SimplicialComplex::new();
    let p0 = Simplex::new(0, vec![0]);
    complex.join_simplex(p0.clone());

    // H_0
    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti number for a point should be 1");
    assert_eq!(h0.homology_generators.len(), 1, "H0: Should have one generator");
    let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, F::one());
    assert!(
      h0.homology_generators.contains(&expected_gen_h0),
      "H0: Generator mismatch for field {:?}",
      std::any::type_name::<F>()
    );

    // H_1
    let h1 = complex.compute_homology::<F>(1);
    assert_eq!(h1.dimension, 1, "H1: Dimension check");
    assert_eq!(h1.betti_number, 0, "H1: Betti number for a point should be 0");
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Should have no generators for field {:?}",
      std::any::type_name::<F>()
    );

    // H_2
    let h2 = complex.compute_homology::<F>(2);
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
    complex.join_simplex(edge01.clone());

    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 1, "H0: Betti for an edge");
    assert_eq!(h0.homology_generators.len(), 1, "H0: One generator");
    let p0 = Simplex::new(0, vec![0]);
    let expected_gen_h0 = Chain::from_simplex_and_coeff(p0, F::one());
    assert!(
      h0.homology_generators.contains(&expected_gen_h0),
      "H0: Generator for edge field {:?}",
      std::any::type_name::<F>()
    );

    let h1 = complex.compute_homology::<F>(1);
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

    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.dimension, 0, "H0: Dimension check");
    assert_eq!(h0.betti_number, 2, "H0: Betti for two points");
    assert_eq!(h0.homology_generators.len(), 2, "H0: Two generators");
    let expected_gen1_h0 = Chain::from_simplex_and_coeff(p0_s, F::one());
    let expected_gen2_h0 = Chain::from_simplex_and_coeff(p1_s, F::one());
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

    let h1 = complex.compute_homology::<F>(1);
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
    complex.join_simplex(triangle012.clone());

    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for triangle field {:?}", std::any::type_name::<F>());

    let h1 = complex.compute_homology::<F>(1);
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

    let h2 = complex.compute_homology::<F>(2);
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

    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti for circle field {:?}", std::any::type_name::<F>());

    let h1 = complex.compute_homology::<F>(1);
    assert_eq!(h1.betti_number, 1, "H1: Betti for circle field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h1.homology_generators.len(),
      1,
      "H1: One 1D generator field {:?}",
      std::any::type_name::<F>()
    );

    let generator_h1 = h1.homology_generators[0].clone();
    assert_eq!(
      generator_h1.simplices.len(),
      3,
      "H1: Circle generator 3 simplices field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.simplices.contains(&s01),
      "H1: Gen missing s01 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.simplices.contains(&s12),
      "H1: Gen missing s12 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h1.simplices.contains(&s02),
      "H1: Gen missing s02 field {:?}",
      std::any::type_name::<F>()
    );
    for coeff in &generator_h1.coefficients {
      assert_eq!(
        *coeff,
        F::one(),
        "H1: Coeffs should be 1 for field {:?}",
        std::any::type_name::<F>()
      );
    }
    assert!(
      generator_h1.boundary().simplices.is_empty(),
      "H1: Generator boundary zero field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.compute_homology::<F>(2);
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

    let h0 = complex.compute_homology::<F>(0);
    assert_eq!(h0.betti_number, 1, "H0: Betti sphere field {:?}", std::any::type_name::<F>());

    let h1 = complex.compute_homology::<F>(1);
    assert_eq!(h1.betti_number, 0, "H1: Betti sphere field {:?}", std::any::type_name::<F>());
    assert!(
      h1.homology_generators.is_empty(),
      "H1: Generators sphere field {:?}",
      std::any::type_name::<F>()
    );

    let h2 = complex.compute_homology::<F>(2);
    assert_eq!(h2.betti_number, 1, "H2: Betti sphere field {:?}", std::any::type_name::<F>());
    assert_eq!(
      h2.homology_generators.len(),
      1,
      "H2: One 2D generator field {:?}",
      std::any::type_name::<F>()
    );

    let generator_h2 = h2.homology_generators[0].clone();
    assert_eq!(
      generator_h2.simplices.len(),
      4,
      "H2: Sphere generator 4 faces field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.simplices.contains(&f012),
      "H2: Gen missing f012 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.simplices.contains(&f013),
      "H2: Gen missing f013 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.simplices.contains(&f023),
      "H2: Gen missing f023 field {:?}",
      std::any::type_name::<F>()
    );
    assert!(
      generator_h2.simplices.contains(&f123),
      "H2: Gen missing f123 field {:?}",
      std::any::type_name::<F>()
    );
    for coeff in &generator_h2.coefficients {
      assert_eq!(
        *coeff,
        F::one(),
        "H2: Coeffs should be 1 for field {:?}",
        std::any::type_name::<F>()
      );
    }
    assert!(
      generator_h2.boundary().simplices.is_empty(),
      "H2: Generator boundary zero field {:?}",
      std::any::type_name::<F>()
    );

    let h3 = complex.compute_homology::<F>(3);
    assert_eq!(h3.betti_number, 0, "H3: Betti sphere field {:?}", std::any::type_name::<F>());
  }

  #[test]
  fn test_homology_sphere_surface_all_fields() {
    test_homology_sphere_surface_generic::<Boolean>();
    test_homology_sphere_surface_generic::<Mod7>();
  }
}
