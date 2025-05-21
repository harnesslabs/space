use std::{collections::HashMap, hash::Hash, mem, ops::Add};

use harness_algebra::{prelude::*, tensors::dynamic::vector::DynamicVector};

use super::*;
use crate::definitions::Topology;

#[derive(Clone, Debug)]
pub struct Chain<'a, T: Topology, R> {
  /// The topological space this chain is part of.
  space:            &'a T,
  /// A vector of objects that are part of this chain.
  pub items:        Vec<T::Item>,
  /// A vector of coefficients of type `R`, corresponding one-to-one with the `simplices`.
  /// `coefficients[i]` is the coefficient for `simplices[i]`.
  pub coefficients: Vec<R>,
}

impl<'a, T: Topology, R: Ring> Chain<'a, T, R> {
  pub const fn new(space: &'a T) -> Self { Self { space, items: vec![], coefficients: vec![] } }

  pub fn from_item_and_coeff(space: &'a T, item: T::Item, coeff: R) -> Self {
    Self { space, items: vec![item], coefficients: vec![coeff] }
  }

  pub fn from_items_and_coeffs(space: &'a T, items: Vec<T::Item>, coeffs: Vec<R>) -> Self {
    Self { space, items, coefficients: coeffs }
  }

  // TODO: Get rid of this method and implement the algebraic operations on chains instead.
  /// Scales the chain by a scalar coefficient.
  /// If the scalar is zero, an empty chain is returned.
  pub fn scaled(self, scalar: R) -> Self
  where R: Copy {
    if scalar.is_zero() {
      return Chain::new(self.space);
    }
    let new_coefficients = self.coefficients.into_iter().map(|c| c * scalar).collect();
    Chain::from_items_and_coeffs(self.space, self.items, new_coefficients)
  }

  pub fn boundary(&self) -> Self
  where
    R: Copy,
    T::Item: PartialEq, {
    let mut total_boundary = Chain::new(self.space);
    for (item, coeff) in self.items.iter().zip(self.coefficients.iter()) {
      let simplex_boundary_chain = self.space.boundary(item);
      let scaled_simplex_boundary = simplex_boundary_chain.scaled(*coeff);
      total_boundary = total_boundary + scaled_simplex_boundary;
    }
    total_boundary
  }

  /// Converts this chain to a coefficient vector in the basis given by the mapping.
  /// The mapping should be from basis elements to their indices in the basis.
  /// The resulting vector will have length equal to the basis size.
  pub fn to_coeff_vector(
    &self,
    basis_map: &HashMap<&T::Item, usize>,
    basis_size: usize,
  ) -> DynamicVector<R>
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
    DynamicVector::new(coeffs)
  }
}

impl<T: Topology, R: PartialEq> PartialEq for Chain<'_, T, R>
where T::Item: PartialEq
{
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

impl<T: Topology, R: Ring> Add for Chain<'_, T, R>
where T::Item: PartialEq
{
  type Output = Self;

  /// Adds two chains together, combining like terms by adding their coefficients.
  ///
  /// The resulting chain will only contain simplices with non-zero coefficients.
  /// Simplices are compared for equality to identify like terms.
  ///
  /// # Type Constraints
  /// * `R` must implement `Add<Output = R>`, `Neg<Output = R>`, `Zero`. (Note: The original doc
  ///   mentioned Clone for R, this version avoids it).
  ///
  /// # Note
  /// This implementation assumes that the input chains might contain simplices of mixed dimensions
  /// or unsorted simplices. It iterates through both chains and combines terms. A more efficient
  /// approach for chains known to be of the same dimension and built from a sorted basis would
  /// use a different strategy.
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

  fn sub(self, other: Self) -> Self::Output { self + (-other) }
}

// TODO: Implement the rest of the algebra traits.
// impl<T,R> LeftModule<R> for Chain<T, R> {
//     type Ring = R;
// }

#[derive(Debug, Clone)]
pub struct Homology<R>
where R: Ring + Copy {
  /// The dimension $k$ for which this homology group $H_k$ is computed.
  pub dimension:           usize,
  /// The Betti number $b_k = \text{rank}(H_k(X; R))$. For field coefficients,
  /// this is the dimension of $H_k$ as a vector space over $R$.
  pub betti_number:        usize,
  /// A basis for the homology group $H_k = Z_k / B_k$.
  /// Each element is a [`Chain<R>`] representing a homology class generator.
  pub homology_generators: Vec<DynamicVector<R>>,
}

impl<R> Homology<R>
where R: Ring + Copy
{
  pub const fn trivial(dimension: usize) -> Self {
    Self { dimension, betti_number: 0, homology_generators: Vec::new() }
  }
}
