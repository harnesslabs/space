//! ## Homology Module
//!
//! This module provides structures and implementations for computing and representing
//! homology groups of topological spaces. Homology is a fundamental concept in
//! algebraic topology, assigning a sequence of algebraic objects (typically abelian
//! groups or modules) to a topological space. These objects, known as homology
//! groups, capture information about the "holes" of different dimensions in the space.
//!
//! ### Key Concepts:
//!
//! - **Chains**: A $k$-chain is a formal sum of $k$-items in a topological space $X$, with
//!   coefficients in a ring $R$. It is an element of the chain group $C_k(X; R)$. Represented by
//!   the [`Chain`] struct.
//!
//! - **Boundary Operator**: The boundary operator $\partial_k: C_k(X; R) \to C_{k-1}(X; R)$ maps a
//!   $k$-chain to a $(k-1)$-chain representing its boundary. A key property is that $\partial_{k-1}
//!   \circ \partial_k = 0$.
//!
//! - **Cycles**: A $k$-cycle is a $k$-chain whose boundary is zero. The set of $k$-cycles forms a
//!   subgroup $Z_k(X; R) = \text{ker}(\partial_k)$ of $C_k(X; R)$.
//!
//! - **Boundaries**: A $k$-boundary is a $k$-chain that is the boundary of some $(k+1)$-chain. The
//!   set of $k$-boundaries forms a subgroup $B_k(X; R) = \text{im}(\partial_{k+1})$ of $C_k(X; R)$.
//!   Since $\partial \circ \partial = 0$, every boundary is a cycle, so $B_k(X; R) \subseteq Z_k(X;
//!   R)$.
//!
//! - **Homology Groups**: The $k$-th homology group $H_k(X; R)$ is defined as the quotient group:
//!   $$ H_k(X; R) = Z_k(X; R) / B_k(X; R) $$ Its elements are equivalence classes of $k$-cycles,
//!   where two cycles are equivalent if their difference is a $k$-boundary.
//!
//! - **Betti Numbers**: When $R$ is a field, $H_k(X; R)$ is a vector space over $R$. Its dimension,
//!   $\beta_k = \text{dim}(H_k(X; R))$, is called the $k$-th Betti number. It counts the number of
//!   $k$-dimensional holes in $X$.
//!
//! This module provides the [`Chain`] struct to represent chains and implements
//! operations like addition, negation, and multiplication by a scalar (coefficient).
//! The [`Homology`] struct is used to store the results of a homology computation,
//! including Betti numbers and generators for the homology groups.
//!
//! The actual computation of homology groups (i.e., finding $Z_k$ and $B_k$ and
//! then the quotient) typically involves constructing chain complexes and computing
//! their homology, often using techniques like Smith Normal Form for integer
//! coefficients or Gaussian elimination for field coefficients. This module focuses on
//! the representation of chains and the final homology results.

use std::{collections::HashMap, hash::Hash, mem, ops::Add};

use cova_algebra::{prelude::*, tensors::dynamic::vector::Vector};

use crate::definitions::Topology;

/// Represents a $k$-chain in a topological space.
///
/// A $k$-chain is a formal sum of $k$-items (often simplices) in a topological space $X$,
/// with coefficients in a ring $R$. It is an element of the chain group $C_k(X; R)$.
///
/// # Type Parameters
///
/// * `'a`: Lifetime parameter for the reference to the topological space.
/// * `T`: The type of the topological space, implementing the [`Topology`] trait.
/// * `R`: The type of the coefficients, typically a [`Ring`].
#[derive(Clone, Debug)]
pub struct Chain<'a, T: Topology, R> {
  /// The topological space this chain is part of.
  pub space:        &'a T,
  /// A vector of topological items (e.g., simplices) that constitute this chain.
  /// Each item corresponds to a term in the formal sum.
  pub items:        Vec<T::Item>,
  /// A vector of coefficients of type `R`, corresponding one-to-one with the `items`.
  /// `coefficients[i]` is the coefficient for `items[i]`.
  pub coefficients: Vec<R>,
}

impl<'a, T: Topology, R: Ring> Chain<'a, T, R> {
  /// Creates a new, empty chain associated with the given topological space.
  ///
  /// The resulting chain has no items and no coefficients.
  ///
  /// # Arguments
  ///
  /// * `space`: A reference to the topological space this chain belongs to.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] instance with empty `items` and `coefficients` vectors.
  pub const fn new(space: &'a T) -> Self { Self { space, items: vec![], coefficients: vec![] } }

  /// Creates a new chain from a single item and its coefficient.
  ///
  /// # Arguments
  ///
  /// * `space`: A reference to the topological space.
  /// * `item`: The topological item (e.g., a simplex).
  /// * `coeff`: The coefficient for the item.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] instance representing `coeff * item`.
  pub fn from_item_and_coeff(space: &'a T, item: T::Item, coeff: R) -> Self {
    Self { space, items: vec![item], coefficients: vec![coeff] }
  }

  /// Creates a new chain from a vector of items and a corresponding vector of coefficients.
  ///
  /// The `items` and `coeffs` vectors must have the same length.
  ///
  /// # Arguments
  ///
  /// * `space`: A reference to the topological space.
  /// * `items`: A vector of topological items.
  /// * `coeffs`: A vector of coefficients, where `coeffs[i]` corresponds to `items[i]`.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] instance.
  ///
  /// # Panics
  ///
  /// This function does not explicitly panic if lengths differ, but behavior is undefined
  /// if `items` and `coeffs` have different lengths when used in subsequent operations.
  /// Consider adding a panic or returning a `Result` in a future revision if this is a concern.
  pub const fn from_items_and_coeffs(space: &'a T, items: Vec<T::Item>, coeffs: Vec<R>) -> Self {
    Self { space, items, coefficients: coeffs }
  }

  /// Computes the boundary of this chain.
  ///
  /// The boundary of a chain $\sum_i c_i \sigma_i$ is defined as $\sum_i c_i \partial(\sigma_i)$,
  /// where $\partial(\sigma_i)$ is the boundary of the item $\sigma_i$ as defined by the
  /// [`Topology::boundary()`] method.
  ///
  /// # Type Constraints
  ///
  /// * `R`: Must be `Copy` to allow coefficients to be scaled.
  /// * `T::Item`: Must be `PartialEq` for combining terms in the resulting boundary chain.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] representing the boundary of `self`.
  pub fn boundary(&self) -> Self
  where
    R: Copy,
    T::Item: PartialEq, {
    let mut total_boundary = Chain::new(self.space);
    for (item, coeff) in self.items.iter().zip(self.coefficients.iter()) {
      let simplex_boundary_chain = self.space.boundary(item);
      let scaled_simplex_boundary = simplex_boundary_chain * *coeff;
      total_boundary = total_boundary + scaled_simplex_boundary;
    }
    total_boundary
  }

  /// Converts this chain to a coefficient vector in a given basis.
  ///
  /// The basis is implicitly defined by `basis_map`, which maps items (basis elements)
  /// to their corresponding indices in the coefficient vector. The `basis_size` determines
  /// the length of the output vector.
  ///
  /// Items in the chain that are not found in `basis_map` are ignored.
  ///
  /// # Arguments
  ///
  /// * `basis_map`: A hash map from references to items (`&T::Item`) to their `usize` indices in
  ///   the basis representation.
  /// * `basis_size`: The total number of elements in the basis, defining the length of the
  ///   resulting vector.
  ///
  /// # Type Constraints
  ///
  /// * `T::Item`: Must be `Hash + Eq` to be used as keys in `basis_map`.
  /// * `R`: Must be `Ring + Copy` for coefficient operations and initialization of the vector.
  ///
  /// # Returns
  ///
  /// A [`Vector<R>`] representing the coefficients of this chain in the specified basis.
  pub fn to_coeff_vector(
    &self,
    basis_map: &HashMap<&T::Item, usize>,
    basis_size: usize,
  ) -> Vector<R>
  where
    T::Item: Hash + Eq,
    R: Ring + Copy,
  {
    let mut coeffs = vec![R::zero(); basis_size];
    for (item, coeff) in self.items.iter().zip(self.coefficients.iter()) {
      if let Some(&idx) = basis_map.get(item) {
        coeffs[idx] = *coeff;
      }
    }
    Vector::new(coeffs)
  }
}

impl<T: Topology, R: PartialEq> PartialEq for Chain<'_, T, R>
where T::Item: PartialEq
{
  /// Checks if two chains are equal.
  ///
  /// Two chains are considered equal if they represent the same formal sum of items.
  /// This means they must have the same set of items, each with the same corresponding
  /// coefficient. The order of items matters in this comparison.
  ///
  /// # Note
  ///
  /// This implementation performs a strict, element-wise comparison of items and
  /// coefficients based on their current order in the respective vectors. For a more
  /// robust equality check that is invariant to the order of terms or handles
  /// chains not in a canonical form (e.g., with zero coefficients or uncombined like
  /// terms), the chains should first be normalized (e.g., sort items, combine
  /// like terms, remove zero-coefficient terms).
  fn eq(&self, other: &Self) -> bool {
    let self_chain = self.coefficients.iter().zip(self.items.iter());
    let other_chain = other.coefficients.iter().zip(other.items.iter());

    self_chain
      .zip(other_chain)
      .all(|((coeff_a, item_a), (coeff_b, item_b))| coeff_a == coeff_b && item_a == item_b)
  }
}

impl<T: Topology, R: Ring> Add for Chain<'_, T, R>
where T::Item: PartialEq
{
  type Output = Self;

  /// Adds two chains together, combining like terms by adding their coefficients.
  ///
  /// The resulting chain will only contain items with non-zero coefficients.
  /// Items are compared for equality to identify like terms.
  ///
  /// # Note
  ///
  /// This implementation iterates through both chains and combines terms. It aims to be
  /// correct for chains that might not be in a canonical form (e.g., unsorted items,
  /// duplicate items before addition). After addition, terms with zero coefficients
  /// are removed.
  fn add(self, other: Self) -> Self::Output {
    let mut result_items = Vec::new();
    let mut result_coefficients: Vec<R> = Vec::new();

    // Process items from self
    for (item_from_self, original_coeff_from_self) in
      self.items.into_iter().zip(self.coefficients.into_iter())
    {
      let mut opt_coeff = Some(original_coeff_from_self);
      let mut found_and_processed = false;

      for (res_idx, res_item_ref) in result_items.iter().enumerate() {
        if item_from_self == *res_item_ref {
          // T: PartialEq
          if let Some(coeff_val) = opt_coeff.take() {
            // Take R from Option, making opt_coeff None
            let current_res_coeff = mem::replace(&mut result_coefficients[res_idx], R::zero());
            result_coefficients[res_idx] = current_res_coeff + coeff_val; // coeff_val is consumed
          }
          found_and_processed = true;
          // item_from_self is owned and will be dropped if not moved later (it won't be in this
          // branch)
          break;
        }
      }

      if !found_and_processed {
        result_items.push(item_from_self); // item_from_self is moved
        if let Some(coeff_val) = opt_coeff.take() {
          // opt_coeff should be Some here
          result_coefficients.push(coeff_val); // coeff_val is moved
        } else {
          // This case should ideally not happen if logic is correct, means opt_coeff was already
          // None. For safety, one might handle it, e.g. by pushing R::zero() or
          // panicking. However, with current flow, if !found_and_processed, opt_coeff
          // must be Some.
        }
      }
      // After this, item_from_self is either dropped (if found_and_processed) or moved (if
      // !found_and_processed). opt_coeff is None because it has been .take()n in one of the
      // branches.
    }

    // Process items from other
    for (item_from_other, original_coeff_from_other) in
      other.items.into_iter().zip(other.coefficients.into_iter())
    {
      let mut opt_coeff = Some(original_coeff_from_other);
      let mut found_and_processed = false;

      for (res_idx, res_item_ref) in result_items.iter().enumerate() {
        if item_from_other == *res_item_ref {
          // T: PartialEq
          if let Some(coeff_val) = opt_coeff.take() {
            let current_res_coeff = mem::replace(&mut result_coefficients[res_idx], R::zero());
            result_coefficients[res_idx] = current_res_coeff + coeff_val;
          }
          found_and_processed = true;
          break;
        }
      }

      if !found_and_processed {
        result_items.push(item_from_other);
        if let Some(coeff_val) = opt_coeff.take() {
          result_coefficients.push(coeff_val);
        } else {
          // Safety: see comment in the loop for self items
        }
      }
    }

    // Filter out zero coefficients
    let mut i = 0;
    while i < result_coefficients.len() {
      if result_coefficients[i].is_zero() {
        // R: Ring implies R: Zero for is_zero()
        result_coefficients.remove(i);
        result_items.remove(i);
      } else {
        i += 1;
      }
    }

    Self { space: self.space, items: result_items, coefficients: result_coefficients }
  }
}

impl<T: Topology, R: Ring + Copy> Neg for Chain<'_, T, R>
where T::Item: PartialEq
{
  type Output = Self;

  /// Negates a chain by negating all its coefficients.
  ///
  /// If the chain is $C = \sum_i c_i \sigma_i$, its negation is $-C = \sum_i (-c_i) \sigma_i$.
  /// The items in the chain remain the same.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] where each coefficient is the negation of the corresponding
  /// coefficient in the original chain.
  fn neg(self) -> Self::Output {
    Self {
      space:        self.space,
      items:        self.items,
      coefficients: self.coefficients.iter().map(|c| -*c).collect(),
    }
  }
}

impl<T: Topology, R: Ring + Copy> Sub for Chain<'_, T, R>
where T::Item: PartialEq
{
  type Output = Self;

  /// Subtracts one chain from another.
  ///
  /// This is equivalent to adding the first chain to the negation of the second chain
  /// ($A - B = A + (-B)$).
  /// Like terms are combined, and terms with zero coefficients are removed from the result.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] representing the difference `self - other`.
  fn sub(self, other: Self) -> Self::Output { self + (-other) }
}

impl<T, R> Mul<R> for Chain<'_, T, R>
where
  T: Topology,
  R: Ring + Copy,
  T::Item: PartialEq,
{
  type Output = Self;

  /// Multiplies a chain by a scalar (coefficient) from the right.
  ///
  /// Each coefficient in the chain is multiplied by the scalar.
  /// If the chain is $C = \sum_i c_i \sigma_i$, and $s$ is a scalar, then $C \cdot s = \sum_i (c_i
  /// \cdot s) \sigma_i$. The items in the chain remain the same.
  ///
  /// If the scalar `other` is zero, the resulting chain will have all zero coefficients.
  /// Note: The current implementation does not automatically remove zero-coefficient terms
  /// that result from this multiplication, unlike `add` or `boundary`. This could be a
  /// point of refinement if a canonical form (no zero terms) is always desired.
  ///
  /// # Arguments
  ///
  /// * `other`: The scalar of type `R` to multiply the chain's coefficients by.
  ///
  /// # Returns
  ///
  /// A new [`Chain`] where each original coefficient has been multiplied by `other`.
  fn mul(self, other: R) -> Self::Output {
    Chain::from_items_and_coeffs(
      self.space,
      self.items,
      self.coefficients.iter().map(|c| *c * other).collect(),
    )
  }
}

/// Represents the $k$-th homology group $H_k(X; R)$ of a topological space $X$
/// with coefficients in a ring $R$.
///
/// It stores the Betti number (rank of the homology group) and a set of generators
/// for the homology group.
///
/// # Type Parameters
///
/// * `R`: The type of the coefficients, which must implement [`Ring`] and `Copy`.
#[derive(Debug, Clone)]
pub struct Homology<R>
where R: Ring + Copy {
  // Note: While struct definition has `Ring + Copy`, individual fields might not always need
  // `Copy` if data is owned.
  /// The dimension $k$ for which this homology group $H_k$ is computed.
  pub dimension:           usize,
  /// The Betti number $\beta_k = \text{rank}(H_k(X; R))$.
  ///
  /// For field coefficients, this is the dimension of $H_k(X; R)$ as a vector space over $R$.
  /// It represents the number of $k$-dimensional "holes" in the space.
  pub betti_number:        usize,
  /// A basis for the homology group $H_k = Z_k / B_k$.
  ///
  /// Each element is a [`Vector<R>`] representing a homology class generator.
  /// These vectors are typically coefficient vectors in some chosen basis for the $k$-cycles.
  pub homology_generators: Vec<Vector<R>>,
}

impl<R> Homology<R>
where R: Ring + Copy
{
  /// Creates a trivial homology group for a given dimension.
  ///
  /// A trivial homology group has a Betti number of 0 and no generators.
  /// This is often used as a placeholder or for dimensions where homology is known to be zero.
  ///
  /// # Arguments
  ///
  /// * `dimension`: The dimension $k$ for this trivial homology group $H_k$.
  ///
  /// # Returns
  ///
  /// A new [`Homology`] instance with `betti_number` set to 0 and an empty
  /// `homology_generators` vector.
  pub const fn trivial(dimension: usize) -> Self {
    Self { dimension, betti_number: 0, homology_generators: Vec::new() }
  }
}
