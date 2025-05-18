use std::ops::Add;

use super::*;

#[derive(Clone, Debug, Default)]
pub struct Chain<T, R> {
  /// A vector of objects that are part of this chain.
  items:        Vec<T>,
  /// A vector of coefficients of type `R`, corresponding one-to-one with the `simplices`.
  /// `coefficients[i]` is the coefficient for `simplices[i]`.
  coefficients: Vec<R>,
}

impl<T, R> Chain<T, R> {
  pub const fn new() -> Self { Self { items: vec![], coefficients: vec![] } }

  pub fn from_items_and_coeffs(items: Vec<T>, coeffs: Vec<R>) -> Self {
    Self { items, coefficients: coeffs }
  }

  /// Computes the boundary of this chain.
  ///
  /// The boundary operator $\partial$ maps a $k$-chain to a $(k-1)$-chain. For a single $k$-simplex
  /// $\sigma = [v_0, v_1, \dots, v_k]$, its boundary is $\partial_k \sigma = \sum_{i=0}^{k} (-1)^i
  /// [v_0, \dots, \hat{v_i}, \dots, v_k]$, where $\hat{v_i}$ indicates that vertex $v_i$ is
  /// omitted. This operator is extended linearly to chains.
  ///
  /// A fundamental property of the boundary operator is that $\partial \circ \partial = 0$ (the
  /// boundary of a boundary is zero).
  ///
  /// # Type Constraints
  /// * `R` must implement `Clone`, `Neg<Output = R>`, `Add<Output = R>`, and `Zero`.
  ///
  /// # Returns
  /// A new [`Chain<R>`] representing the boundary of the current chain. If the chain consists of
  /// 0-simplices, their boundary is the empty chain (zero).
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

impl<T: PartialEq, R: PartialEq> PartialEq for Chain<T, R> {
  /// Checks if two chains are equal.
  ///
  /// Two chains are considered equal if they represent the same formal sum of simplices.
  /// This means they must have the same set of simplices, each with the same corresponding
  /// coefficient. This implementation also considers the orientation of simplices (via
  /// `permutation_sign`) and the equality of their vertex lists, which might be overly strict or
  /// not standard depending on the context (usually, chains are simplified so simplices are
  /// unique and then coefficients are compared).
  ///
  /// **Note:** The current equality check based on `permutation_sign` and exact order in zipped
  /// iterators might be problematic if chains are not in a canonical form (e.g., sorted
  /// simplices, combined like terms). A more robust equality would typically involve converting
  /// both chains to a canonical representation first.
  fn eq(&self, other: &Self) -> bool {
    let self_chain = self.coefficients.iter().zip(self.items.iter());
    let other_chain = other.coefficients.iter().zip(other.items.iter());

    self_chain
      .zip(other_chain)
      .all(|((coeff_a, item_a), (coeff_b, item_b))| coeff_a == coeff_b && item_a == item_b)
  }
}

impl<T: PartialEq, R: Ring> Add for Chain<T, R> {
  type Output = Self;

  /// Adds two chains together, combining like terms by adding their coefficients.
  ///
  /// The resulting chain will only contain simplices with non-zero coefficients.
  /// Simplices are compared for equality to identify like terms.
  ///
  /// # Type Constraints
  /// * `R` must implement `Add<Output = R>`, `Neg<Output = R>`, `Clone`, and `Zero`.
  ///
  /// # Note
  /// This implementation assumes that the input chains might contain simplices of mixed dimensions
  /// or unsorted simplices. It iterates through both chains and combines terms. A more efficient
  /// approach for chains known to be of the same dimension and built from a sorted basis would
  /// use a different strategy.
  fn add(self, other: Self) -> Self::Output {
    let mut result_items = Vec::new();
    let mut result_coefficients: Vec<R> = Vec::new();

    // Add all simplices from self
    for (idx, item) in self.items.into_iter().enumerate() {
      let coefficient = self.coefficients[idx];

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, &res_item) in result_items.iter().enumerate() {
        if item == res_item {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx] + coefficient;
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_items.push(item);
        result_coefficients.push(coefficient);
      }
    }

    // Add all simplices from other
    for (idx, item) in other.items.into_iter().enumerate() {
      let coefficient = other.coefficients[idx];

      // See if this simplex already exists in our result
      let mut found = false;
      for (res_idx, &res_item) in result_items.iter().enumerate() {
        if item == res_item {
          // Same simplex, add coefficients
          result_coefficients[res_idx] = result_coefficients[res_idx] + coefficient;
          found = true;
          break;
        }
      }

      if !found {
        // New simplex
        result_items.push(item);
        result_coefficients.push(coefficient);
      }
    }

    // Filter out zero coefficients
    let mut i = 0;
    while i < result_coefficients.len() {
      if result_coefficients[i].is_zero() {
        result_coefficients.remove(i);
        result_items.remove(i);
      } else {
        i += 1;
      }
    }

    Self { items: result_items, coefficients: result_coefficients }
  }
}

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

// TODO: The `HomologyGroup` struct mentions coefficients `R: Field`.
// While computations often use fields (like Z/2Z or Q or R), homology can be defined over rings
// (like Z). The current implementation heavily relies on field properties for Gaussian elimination.
/// Stores the results of a homology computation for a specific dimension $k$.
/// This includes the Betti number (rank of $H_k$) and bases for cycles, boundaries, and $H_k$
/// itself.
///
/// # Type Parameters
/// * `R`: The coefficient type, which **must be a field** for the current computation methods (due
///   to reliance on Gaussian elimination which requires multiplicative inverses). It should
///   implement [`Field`].
#[derive(Debug, Clone)]
pub struct HomologyGroup<R: Field> {
  /// The dimension $k$ for which this homology group $H_k$ is computed.
  pub dimension:           usize,
  /// The Betti number $b_k = \text{rank}(H_k(X; R))$. For field coefficients,
  /// this is the dimension of $H_k$ as a vector space over $R$.
  pub betti_number:        usize,
  /// A basis for the $k$-cycles $Z_k = \ker \partial_k$.
  /// Each element is a [`Chain<R>`] representing a cycle.
  pub cycle_generators:    Vec<Chain<R>>,
  /// A basis for the $k$-boundaries $B_k = \text{im } \partial_{k+1}$.
  /// Each element is a [`Chain<R>`] representing a boundary.
  pub boundary_generators: Vec<Chain<R>>,
  /// A basis for the homology group $H_k = Z_k / B_k$.
  /// Each element is a [`Chain<R>`] representing a homology class generator.
  pub homology_generators: Vec<Chain<R>>,
}

impl<R: Field> HomologyGroup<R> {
  /// Creates a new, trivial homology group for a given `dimension`.
  /// A trivial group has Betti number 0 and no generators.
  /// This is used when, for example, there are no $k$-simplices to form $k$-chains.
  pub const fn trivial(dimension: usize) -> Self {
    Self {
      dimension,
      betti_number: 0,
      cycle_generators: Vec::new(),
      boundary_generators: Vec::new(),
      homology_generators: Vec::new(),
    }
  }
}
