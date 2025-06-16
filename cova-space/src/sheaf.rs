//! # Sheaf Module
//!
//! This module defines the structure and operations for a **sheaf** over a topological space.
//! In mathematics, a sheaf is a tool for systematically tracking locally defined data
//! attached to the open sets of a topological space and defined from the open sets to
//! a category.
//!
//! ## Core Concepts
//!
//! - **Topological Space ($T$)**: The base space over which the sheaf is defined. In this context,
//!   the space is also a [`Poset`] (partially ordered set), where the items of the topology (e.g.,
//!   cells in a cell complex, open sets) are ordered by inclusion or a similar relation. The `Item`
//!   type of the `Topology` trait represents these fundamental components (e.g., cells).
//!
//! - **Category ($\mathcal{C}$)**: A target category where the data (stalks or sections) of the
//!   sheaf resides. For each item $U$ in the topological space $T$, a sheaf assigns an object
//!   $F(U)$ from $\mathcal{C}$. The type `C` in `Sheaf<T, C>` represents objects in this category,
//!   and `C::Morphism` represents the morphisms (e.g., functions, linear maps).
//!
//! - **Restriction Maps**: For any pair of items $U, V$ in $T$ such that $U \subseteq V$ (or more
//!   generally $U \le V$ in the poset structure), there is a **restriction morphism**
//!   $\text{res}_{U,V}: F(V) \to F(U)$. These morphisms must satisfy certain compatibility
//!   conditions:
//!   1. $\text{res}_{U,U} = \text{id}_{F(U)}$ (identity on $U$).
//!   2. If $W \subseteq V \subseteq U$ (or $W \le V \le U$), then $\text{res}_{W,V} \circ
//!      \text{res}_{V,U} = \text{res}_{W,U}$ (composition).
//!
//! - **Sections**: A section $s$ over an item $U \in T$ is an element $s \in F(U)$. A **global
//!   section** is a collection of sections $\{s_U \in F(U)\}$ for each $U \in T$ such that for any
//!   pair $U \le V$, $\text{res}_{U,V}(s_V) = s_U$.
//!
//! The [`Sheaf`] struct in this module stores the underlying topological space (which is also a
//! poset) and the restriction maps between its items.
//!
//! ## Implementation Details
//!
//! - The `space` field holds the topological space $T$, which must implement both [`Topology`] and
//!   [`Poset`]. Its items `T::Item` must be `Hash + Eq + Clone` to be used as keys in HashMaps.
//! - The `restrictions` field is a `HashMap`. For a key `(U, V)` where `U` is a \"parent\" item and
//!   `V` is a \"child\" item (meaning $U \le V$ in the poset structure of the space), the
//!   associated value `C::Morphism` is the restriction map $\rho_{UV}: F(V) \to F(U)$.
//! - The `Category` trait (`C`) provides the structure for the data attached to items, including
//!   the `Morphism` type and an `apply` method to apply morphisms to objects.

use std::{collections::HashMap, fmt::Debug, hash::Hash};

use cova_algebra::tensors::{DMatrix, DVector};

use super::*;
use crate::{
  complexes::{Complex, ComplexElement},
  definitions::Topology,
  set::Poset,
};

// TODO: We should make this have a nice construction setup so you can build the underlying space
// and the restrictions simultaneously

/// Represents a sheaf over a topological space `T` with values in a category `C`.
///
/// A sheaf assigns an object from a category `C` to each item (e.g., open set, cell)
/// in a topological space `T`. It also defines restriction morphisms that relate
/// the data assigned to different items, ensuring consistency.
///
/// # Type Parameters
///
/// * `T`: The underlying topological space, which must also implement [`Poset`]. Its items
///   (`T::Item`) must be `Hash + Eq + Clone` for use in `HashMap` keys.
/// * `C`: The target [`Category`] where the data (stalks/sections) of the sheaf reside. Objects of
///   this category (`C`) must be `Clone + Eq + Debug` (if `is_global_section` is used). Morphisms
///   (`C::Morphism`) must be `Clone + Debug`.
pub struct Sheaf<T: Topology, C: Category> {
  /// The underlying topological space (e.g., a cell complex, an ordered set of open sets).
  /// This space also implements [`Poset`] to define relationships (e.g., sub-item/super-item)
  /// between its items.
  pub space:        T,
  /// A map defining the restriction morphisms of the sheaf.
  ///
  /// For a key `(parent_item, child_item)` where `parent_item <= child_item` according to the
  /// poset structure of `T`, the associated value `C::Morphism` is the restriction map
  /// from the data on `child_item` to the data on `parent_item`.
  /// That is, $\rho_{\text{parent_item}, \text{child_item}}: F(\text{child_item}) \to
  /// F(\text{parent_item})$.
  pub restrictions: HashMap<(T::Item, T::Item), C::Morphism>,
}

impl<T, C> Sheaf<T, C>
where
  T: Topology + Poset,
  T::Item: Hash + Eq + Clone + Debug,
  C: Category + Clone + PartialEq + Debug,
  C::Morphism: Clone + Debug,
{
  /// Creates a new sheaf from a given topological space and a set of restriction maps.
  ///
  /// The provided `restrictions` map should contain morphisms for all relevant pairs
  /// of items $(U, V)$ in the space `T` where $U \le V$. The constructor asserts that
  /// for every key `(k.0, k.1)` in `restrictions`, the relation `space.leq(&k.0, &k.1)` holds.
  /// Here, `k.0` is treated as the parent (smaller item) and `k.1` as the child (larger item).
  /// The restriction map is from $F(k.1)$ to $F(k.0)$.
  ///
  /// # Arguments
  ///
  /// * `space`: The topological space `T` which also implements `Poset`.
  /// * `restrictions`: A `HashMap` where keys are `(parent_item, child_item)` tuples from `T::Item`
  ///   (such that `parent_item <= child_item`), and values are the corresponding restriction
  ///   morphisms $F(\text{child_item}) \to F(\text{parent_item})$ of type `C::Morphism`.
  ///
  /// # Panics
  ///
  /// * Panics if any key `(parent, child)` in `restrictions` does not satisfy `space.leq(&parent,
  ///   &child) == Some(true)`. This includes cases where `space.leq` returns `Some(false)`
  ///   (relation does not hold) or `None` (incomparable).
  ///
  /// # TODO
  /// - Implement a builder API for more robust construction, ensuring all necessary restrictions
  ///   are defined (e.g., for all successor relationships in the poset, and identity maps for $U
  ///   \le U$).
  /// - Add validation for compatibility of restriction maps (e.g., identity $\text{res}_{U,U} =
  ///   \text{id}_{F(U)}$ and composition $\text{res}_{W,V} \circ \text{res}_{V,U} =
  ///   \text{res}_{W,U}$ axioms).
  /// - For specific categories (e.g., matrices as morphisms), check dimensional compatibility of
  ///   morphisms.
  pub fn new(space: T, restrictions: HashMap<(T::Item, T::Item), C::Morphism>) -> Self {
    assert!(
      restrictions.iter().all(|(k, _v)| space.leq(&k.0, &k.1) == Some(true)),
      "Restriction map defined for a pair (parent, child) where parent is not less than or equal \
       to child, or they are incomparable."
    );

    Self { space, restrictions }
  }

  /// Restricts data from a larger item to a smaller item using the sheaf's restriction map.
  ///
  /// Given a `parent_target_item` $U$ and a `child_source_item` $V$ such that $U \le V$,
  /// this function retrieves the restriction morphism $\rho_{UV}: F(V) \to F(U)$ from the sheaf's
  /// definition (stored under the key `(parent_target_item, child_source_item)`).
  /// It then retrieves the data $s_V \in F(V)$ corresponding to `child_source_item` from the
  /// provided `section` data. Finally, it applies the morphism to compute $s_U = \rho_{UV}(s_V)$,
  /// which is the data on $U$ restricted from $V$.
  ///
  /// # Arguments
  ///
  /// * `parent_target_item`: The smaller item $U$ to which data is being restricted.
  /// * `child_source_item`: The larger item $V$ from which data is being restricted. Must satisfy
  ///   `parent_target_item <= child_source_item`.
  /// * `section_data_on_child`: The data object $s_V \in F(V)$ associated with `child_source_item`.
  ///
  /// # Returns
  ///
  /// The restricted data $s_U \in F(U)$, result of applying the restriction map.
  ///
  /// # Panics
  ///
  /// * Panics if `parent_target_item <= child_source_item` is false or if they are incomparable, as
  ///   asserted by `self.space.leq`.
  /// * Panics if the restriction map for `(parent_target_item, child_source_item)` is not found in
  ///   `self.restrictions`.
  pub fn restrict(
    &self,
    parent_target_item: &T::Item,
    child_source_item: &T::Item,
    section_data_on_child: C,
  ) -> C {
    assert!(
      self.space.leq(parent_target_item, child_source_item) == Some(true),
      "Cannot restrict: parent_target_item is not less than or equal to child_source_item, or \
       items are incomparable."
    );

    let restriction_map = self
      .restrictions
      .get(&(parent_target_item.clone(), child_source_item.clone()))
      .unwrap_or_else(|| {
        panic!("Restriction map not found for ({parent_target_item:?}, {child_source_item:?})")
      });

    C::apply(restriction_map.clone(), section_data_on_child)
  }

  /// Checks if a given `section` is a global section of the sheaf.
  ///
  /// A section $s = \{s_X\}_{X \in T}$ is a global section if for every pair of items
  /// `(parent_item, child_item)` in `self.restrictions` (which implies `parent_item <=
  /// child_item`), the restriction of the data on `child_item` to `parent_item` is equal to the
  /// data already on `parent_item`.
  ///
  /// That is, for each stored morphism $\rho_{\text{parent}, \text{child}}: F(\text{child_item})
  /// \to F(\text{parent_item})$, it must hold that $\rho_{\text{parent},
  /// \text{child}}(s_{\text{child_item}}) = s_{\text{parent_item}}$, where $s_{\
  /// text{child_item}}$ is `section.get(child_item)` and $s_{\text{parent_item}}$ is
  /// `section.get(parent_item)`.
  ///
  /// # Arguments
  ///
  /// * `section`: A `HashMap` where keys are items `T::Item` from the space and values are data
  ///   objects of type `C` (from the target category), representing $s_X$ for each $X$.
  ///
  /// # Returns
  ///
  /// * `true` if the `section` satisfies the global section condition for all defined restrictions.
  /// * `false` if any restriction condition is violated, or if data for a required item (involved
  ///   in a restriction) is missing from the `section`.
  pub fn is_global_section(&self, section: &HashMap<T::Item, C>) -> bool {
    for ((parent_item, child_item), restriction_map) in &self.restrictions {
      let Some(parent_data) = section.get(parent_item) else {
        return false;
      };
      let Some(child_data) = section.get(child_item) else {
        return false;
      };
      let restricted_parent_data = C::apply(restriction_map.clone(), parent_data.clone());
      if restricted_parent_data != *child_data {
        return false;
      }
    }
    true
  }
}

// TODO: This is a temporary implementation for the coboundary map specifically for the vector
// stalks.
impl<T: ComplexElement, F: Field + Copy> Sheaf<Complex<T>, DVector<F>>
where T: Hash + Eq + Clone + Debug
{
  /// Constructs the coboundary matrix δ^k: C^k → C^(k+1) for the sheaf.
  ///
  /// The coboundary map is dual to the boundary map of the underlying complex.
  /// For dimension k, this maps k-cochains (sections over k-dimensional elements)
  /// to (k+1)-cochains (sections over (k+1)-dimensional elements).
  ///
  /// The matrix has:
  /// - Rows indexed by (k+1)-dimensional elements
  /// - Columns indexed by k-dimensional elements
  /// - Entry (σ, τ) equals the orientation coefficient of τ in ∂σ where σ is (k+1)-dimensional and
  ///   τ is k-dimensional
  ///
  /// # Arguments
  /// * `dimension`: The dimension k of the domain (k-cochains)
  ///
  /// # Returns
  /// A matrix representing δ^k: C^k → C^(k+1)
  pub fn coboundary(&self, dimension: usize) -> DMatrix<F> {
    // Get sorted k-dimensional and (k+1)-dimensional elements
    let k_elements = {
      let mut elements = self.space.elements_of_dimension(dimension);
      elements.sort_unstable();
      elements
    };

    let k_plus_1_elements = {
      let mut elements = self.space.elements_of_dimension(dimension + 1);
      elements.sort_unstable();
      elements
    };

    if k_elements.is_empty() || k_plus_1_elements.is_empty() {
      // No source elements or no target elements - return empty matrix
      return DMatrix::<F>::zeros(0, 0);
    }

    // Determine block sizes based on stalk dimensions
    let mut col_block_sizes = Vec::new();
    for k_element in &k_elements {
      // Find any restriction involving this k_element to determine its stalk dimension
      let stalk_dim = self
        .restrictions
        .iter()
        .find_map(|((from, to), matrix)| {
          if from.same_content(k_element) {
            Some(matrix.ncols()) // When k_element is the source, its stalk dimension is num_cols
          } else if to.same_content(k_element) {
            Some(matrix.nrows()) // When k_element is the target, its stalk dimension is num_rows
          } else {
            None
          }
        })
        .unwrap_or(1); // Default to 1 if no restriction found
      col_block_sizes.push(stalk_dim);
    }

    let mut row_block_sizes = Vec::new();
    for k_plus_1_element in &k_plus_1_elements {
      // Find any restriction involving this (k+1)-element to determine its stalk dimension
      let stalk_dim = self
        .restrictions
        .iter()
        .find_map(|((from, to), matrix)| {
          if from.same_content(k_plus_1_element) {
            Some(matrix.ncols()) // When k_plus_1_element is the source, its stalk dimension is
                                 // num_cols
          } else if to.same_content(k_plus_1_element) {
            Some(matrix.nrows()) // When k_plus_1_element is the target, its stalk dimension is
                                 // num_rows
          } else {
            None
          }
        })
        .unwrap_or(1); // Default to 1 if no restriction found
      row_block_sizes.push(stalk_dim);
    }

    // Pre-compute offsets for each block row/column.
    let row_offsets: Vec<usize> = {
      let mut offs = Vec::with_capacity(row_block_sizes.len());
      let mut acc = 0usize;
      for &size in &row_block_sizes {
        offs.push(acc);
        acc += size;
      }
      offs
    };
    let col_offsets: Vec<usize> = {
      let mut offs = Vec::with_capacity(col_block_sizes.len());
      let mut acc = 0usize;
      for &size in &col_block_sizes {
        offs.push(acc);
        acc += size;
      }
      offs
    };

    let total_rows: usize = row_block_sizes.iter().sum();
    let total_cols: usize = col_block_sizes.iter().sum();
    let mut result = DMatrix::<F>::zeros(total_rows, total_cols);

    for (row_idx, k_plus_1_element) in k_plus_1_elements.iter().enumerate() {
      for (col_idx, k_element) in k_elements.iter().enumerate() {
        // Determine if k_element is in boundary
        let boundary_with_orientations = k_plus_1_element.boundary_with_orientations();
        if let Some((_, orientation_coeff)) =
          boundary_with_orientations.iter().find(|(face, _)| face.same_content(k_element))
        {
          if let Some(restriction_matrix) =
            self.restrictions.get(&(k_element.clone(), k_plus_1_element.clone()))
          {
            // Signed matrix
            let mut signed = restriction_matrix.clone();
            if *orientation_coeff < 0 {
              for val in signed.iter_mut() {
                *val = -*val;
              }
            } else if *orientation_coeff == 0 {
              signed.fill(F::zero());
            }

            // Place into result
            let row_offset = row_offsets[row_idx];
            let col_offset = col_offsets[col_idx];
            let r_rows = signed.nrows();
            let r_cols = signed.ncols();
            result.view_mut((row_offset, col_offset), (r_rows, r_cols)).copy_from(&signed);
          }
        }
      }
    }

    result
  }
}

#[cfg(test)]
mod tests {
  #![allow(clippy::type_complexity)]
  #![allow(clippy::too_many_lines)]
  #![allow(clippy::float_cmp)]
  use cova_algebra::tensors::{DMatrix, DVector, MatrixBuilder};

  use super::*;
  use crate::complexes::{Cube, CubicalComplex, Simplex, SimplicialComplex};

  fn simplicial_complex_1d(
  ) -> (SimplicialComplex, HashMap<(Simplex, Simplex), DMatrix<f64>>, Simplex, Simplex, Simplex) {
    let mut cc = SimplicialComplex::new();
    let v0 = Simplex::new(0, vec![0]);
    let v1 = Simplex::new(0, vec![1]);
    let e01 = Simplex::new(1, vec![0, 1]);
    let v0 = cc.join_element(v0);
    let v1 = cc.join_element(v1);
    let e01 = cc.join_element(e01);
    let restrictions = HashMap::from([
      ((v0.clone(), e01.clone()), { MatrixBuilder::new().column([1.0, 2.0]).build() }),
      ((v1.clone(), e01.clone()), {
        MatrixBuilder::new().column([2.0, 0.0]).column([0.0, 2.0]).build()
      }),
    ]);
    (cc, restrictions, v0, v1, e01)
  }

  #[test]
  fn test_simplicial_sheaf_global_section_1d() {
    let (cc, restrictions, v1, v2, e1) = simplicial_complex_1d();

    let sheaf = Sheaf::<SimplicialComplex, DVector<f64>>::new(cc, restrictions);

    let section = HashMap::from([
      (v1.clone(), DVector::from_row_slice(&[2.0])), // R^1
      (v2.clone(), DVector::from_row_slice(&[1.0, 2.0])), // R^2
      (e1.clone(), DVector::from_row_slice(&[2.0, 4.0])), // R^2
    ]);
    assert!(sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1.clone(), DVector::from_row_slice(&[1.0])), // R^1
      (v2.clone(), DVector::from_row_slice(&[1.0, 2.0])), // R^2
      (e1.clone(), DVector::from_row_slice(&[2.0, 4.0])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1.clone(), DVector::from_row_slice(&[2.0])), // R^1
      (v2.clone(), DVector::from_row_slice(&[1.0, 2.0])), // R^2
      (e1.clone(), DVector::from_row_slice(&[1.0, 2.0])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1, DVector::from_row_slice(&[2.0])),      // R^1
      (v2, DVector::from_row_slice(&[3.0, 3.0])), // R^2
      (e1, DVector::from_row_slice(&[2.0, 4.0])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));
  }

  #[test]
  fn test_simplicial_sheaf_coboundary_1d() {
    let (cc, restrictions, ..) = simplicial_complex_1d();
    let sheaf = Sheaf::<SimplicialComplex, DVector<f64>>::new(cc, restrictions);
    let coboundary = sheaf.coboundary(0);

    // Expected structure: 1 block row × 2 block columns
    // Block (0,0): 2×1 (from v0's R¹ stalk to e01's R² stalk)
    // Block (0,1): 2×2 (from v1's R² stalk to e01's R² stalk)
    assert_eq!(coboundary.nrows(), 2); // 2 rows (e01 stalk), 3 cols (v0 + v1 stalks)
    assert_eq!(coboundary.ncols(), 3);

    // Extract blocks manually for testing
    let block_00 = coboundary.view((0, 0), (2, 1)); // 2×1 block
    let block_01 = coboundary.view((0, 1), (2, 2)); // 2×2 block

    // Block (0,0): Should be -1 × [[1.0], [2.0]] = [[-1.0], [-2.0]]
    // (since v0 has orientation coefficient -1 in ∂e01 = v1 - v0)
    assert_eq!(block_00.nrows(), 2);
    assert_eq!(block_00.ncols(), 1);
    assert_eq!(block_00[(0, 0)], -1.0);
    assert_eq!(block_00[(1, 0)], -2.0);

    // Block (0,1): Should be +1 × [[2.0, 0.0], [0.0, 2.0]] = [[2.0, 0.0], [0.0, 2.0]]
    // (since v1 has orientation coefficient +1 in ∂e01 = v1 - v0)
    assert_eq!(block_01.nrows(), 2);
    assert_eq!(block_01.ncols(), 2);
    assert_eq!(block_01[(0, 0)], 2.0);
    assert_eq!(block_01[(0, 1)], 0.0);
    assert_eq!(block_01[(1, 0)], 0.0);
    assert_eq!(block_01[(1, 1)], 2.0);

    // Verify the full matrix is correct
    assert_eq!(coboundary.nrows(), 2);
    assert_eq!(coboundary.ncols(), 3);

    // Row 0: [-1, 2, 0]
    assert_eq!(coboundary[(0, 0)], -1.0);
    assert_eq!(coboundary[(0, 1)], 2.0);
    assert_eq!(coboundary[(0, 2)], 0.0);

    // Row 1: [-2, 0, 2]
    assert_eq!(coboundary[(1, 0)], -2.0);
    assert_eq!(coboundary[(1, 1)], 0.0);
    assert_eq!(coboundary[(1, 2)], 2.0);

    println!("Coboundary matrix:");
    println!("{coboundary}");

    let coboundary = sheaf.coboundary(1);
    println!("{coboundary}");
    assert!(coboundary.is_empty()); // No 2-dimensional elements
  }

  fn simplicial_complex_2d() -> (
    SimplicialComplex,
    HashMap<(Simplex, Simplex), DMatrix<f64>>,
    Simplex,
    Simplex,
    Simplex,
    Simplex,
    Simplex,
    Simplex,
    Simplex,
  ) {
    let mut cc = SimplicialComplex::new();
    // Vertices
    let v0 = Simplex::new(0, vec![0]); // R^1
    let v1 = Simplex::new(0, vec![1]); // R^2
    let v2 = Simplex::new(0, vec![2]); // R^3
                                       // Edges
    let e01 = Simplex::new(1, vec![0, 1]); // R^2
    let e02 = Simplex::new(1, vec![0, 2]); // R^2
    let e12 = Simplex::new(1, vec![1, 2]); // R^2

    // Faces
    let f012 = Simplex::new(2, vec![0, 1, 2]); // R^3

    let v0 = cc.join_element(v0);
    let v1 = cc.join_element(v1);
    let v2 = cc.join_element(v2);
    let e01 = cc.join_element(e01);
    let e02 = cc.join_element(e02);
    let e12 = cc.join_element(e12);
    let f012 = cc.join_element(f012);

    let restrictions = HashMap::from([
      ((v0.clone(), e01.clone()), { MatrixBuilder::new().column([1.0, 2.0]).build() }),
      ((v1.clone(), e01.clone()), {
        MatrixBuilder::new().column([1.0, 0.0]).column([0.0, 1.0]).build()
      }),
      ((v0.clone(), e02.clone()), { MatrixBuilder::new().column([1.0, 0.0]).build() }),
      ((v2.clone(), e02.clone()), {
        MatrixBuilder::new().column([1.0, 0.0]).column([0.0, 0.0]).column([0.0, 0.0]).build()
      }),
      ((v1.clone(), e12.clone()), {
        MatrixBuilder::new().column([2.0, 0.0]).column([0.0, 2.0]).build()
      }),
      ((v2.clone(), e12.clone()), {
        MatrixBuilder::new().column([2.0, 0.0]).column([0.0, 2.0]).column([0.0, 0.0]).build()
      }),
      ((e01.clone(), f012.clone()), {
        MatrixBuilder::new().column([2.0, 0.0, 0.0]).column([0.0, 0.0, 0.0]).build()
      }),
      ((e02.clone(), f012.clone()), {
        MatrixBuilder::new().column([2.0, 0.0, 0.0]).column([0.0, 1.0, 0.0]).build()
      }),
      ((e12.clone(), f012.clone()), {
        MatrixBuilder::new().column([1.0, 0.0, 0.0]).column([0.0, 0.0, 0.0]).build()
      }),
    ]);
    (cc, restrictions, v0, v1, v2, e01, e02, e12, f012)
  }

  #[test]
  fn test_simplicial_sheaf_global_section_2d() {
    let (cc, restrictions, v0, v1, v2, e01, e02, e12, f012) = simplicial_complex_2d();

    let sheaf = Sheaf::<SimplicialComplex, DVector<f64>>::new(cc, restrictions);

    let section = HashMap::from([
      (v0, DVector::from_row_slice(&[1.0])),             // R^1
      (v1, DVector::from_row_slice(&[1.0, 2.0])),        // R^2
      (v2, DVector::from_row_slice(&[1.0, 2.0, 3.0])),   // R^3
      (e01, DVector::from_row_slice(&[1.0, 2.0])),       // R^2
      (e02, DVector::from_row_slice(&[1.0, 0.0])),       // R^2
      (e12, DVector::from_row_slice(&[2.0, 4.0])),       // R^2
      (f012, DVector::from_row_slice(&[2.0, 0.0, 0.0])), // R^3
    ]);
    assert!(sheaf.is_global_section(&section));
  }

  #[test]
  fn test_simplicial_sheaf_coboundary_2d() {
    let (cc, restrictions, ..) = simplicial_complex_2d();
    let sheaf = Sheaf::<SimplicialComplex, DVector<f64>>::new(cc, restrictions);
    let coboundary = sheaf.coboundary(0);
    println!("{coboundary}");
    assert_eq!(coboundary.nrows(), 6); // 3 edges with 2×2 stalks = 6×6
    assert_eq!(coboundary.ncols(), 6);

    let coboundary = sheaf.coboundary(1);
    println!("{coboundary}");
    assert_eq!(coboundary.nrows(), 3); // 1 face with 3×3 stalk, 3 edges with 2×2 stalks
    assert_eq!(coboundary.ncols(), 6);

    let coboundary = sheaf.coboundary(2);
    println!("{coboundary}");
    assert!(coboundary.is_empty()); // No 3-dimensional elements
  }

  fn cubical_complex_2d() -> (CubicalComplex, HashMap<(Cube, Cube), DMatrix<f64>>) {
    let mut cc = CubicalComplex::new();

    // Create a 2x2 grid of cubes
    // Vertices (0-cubes) at grid positions
    let v00 = Cube::vertex(0);
    let v10 = Cube::vertex(1);
    let v01 = Cube::vertex(2);
    let v11 = Cube::vertex(3);

    // Horizontal edges (1-cubes)
    let e_h1 = Cube::edge(0, 1);
    let e_h2 = Cube::edge(2, 3);

    // Vertical edges (1-cubes)
    let e_v1 = Cube::edge(0, 2);
    let e_v2 = Cube::edge(1, 3);

    // Square face (2-cube)
    let square = Cube::square([0, 1, 2, 3]); // R^4 stalk

    // Add all elements to complex
    let v00 = cc.join_element(v00); // R^2
    let v10 = cc.join_element(v10); // R^2
    let v01 = cc.join_element(v01); // R^3
    let v11 = cc.join_element(v11); // R^3
    let e_h1 = cc.join_element(e_h1); // R^2
    let e_h2 = cc.join_element(e_h2); // R^3
    let e_v1 = cc.join_element(e_v1); // R^2
    let e_v2 = cc.join_element(e_v2); // R^3
    let square = cc.join_element(square); // R^4

    let restrictions = HashMap::from([
      ((v00.clone(), e_h1.clone()), {
        MatrixBuilder::new().row([1.0, 0.5]).row([0.0, 0.0]).build() // R^2 → R^2
      }),
      ((v10.clone(), e_h1.clone()), {
        MatrixBuilder::new().row([1.0, 0.0]).row([0.0, 1.0]).build() // R^2 → R^2
      }),
      ((v01.clone(), e_h2.clone()), {
        MatrixBuilder::new().row([1.0, 0.0, 0.0]).row([0.0, 0.0, 0.0]).row([0.0, 0.0, 0.0]).build() // R^3 → R^3
      }),
      ((v11.clone(), e_h2.clone()), {
        MatrixBuilder::new().row([0.0, 1.0, 0.0]).row([0.0, 0.0, 1.0]).row([1.0, 0.0, 0.0]).build() // R^3 → R^3
      }),
      ((v00, e_v1.clone()), {
        MatrixBuilder::new().row([2.0, 1.0]).row([0.0, 0.0]).build() // R^2 → R^2
      }),
      ((v01, e_v1.clone()), {
        MatrixBuilder::new().row([1.0, 0.0, 0.0]).row([0.0, 0.0, 0.0]).build() // R^3 → R^2
      }),
      ((v10, e_v2.clone()), {
        MatrixBuilder::new().row([1.0, 0.0]).row([0.0, 1.0]).row([0.0, 0.0]).build() // R^2 → R^3
      }),
      ((v11, e_v2.clone()), {
        MatrixBuilder::new().row([1.0, 0.0, 0.0]).row([0.0, 1.0, 0.0]).row([0.0, 0.0, 0.0]).build() // R^3 → R^3
      }),
      ((e_h1, square.clone()), {
        MatrixBuilder::new().row([1.0, 0.0]).row([0.0, 1.0]).row([0.0, 0.0]).row([0.0, 0.0]).build()
        // R^2 → R^4
      }),
      ((e_h2, square.clone()), {
        MatrixBuilder::new()
          .row([0.0, 0.0, 1.0])
          .row([0.0, 0.0, 0.0])
          .row([0.0, 0.0, 0.0])
          .row([1.0, 0.0, 0.0])
          .build() // R^3 → R^4
      }),
      ((e_v1, square.clone()), {
        MatrixBuilder::new().row([1.0, 0.0]).row([0.0, 0.0]).row([0.0, 0.0]).row([0.0, 0.0]).build()
        // R^2 → R^4
      }),
      ((e_v2, square), {
        MatrixBuilder::new()
          .row([0.0, 1.0, 0.0])
          .row([0.0, 0.0, 0.0])
          .row([0.0, 0.0, 0.0])
          .row([0.0, 0.0, 0.0])
          .build() // R^3 → R^4
      }),
    ]);

    (cc, restrictions)
  }

  #[test]
  fn test_cubical_sheaf_coboundary_2d() {
    let (cc, restrictions) = cubical_complex_2d();
    let sheaf = Sheaf::<CubicalComplex, DVector<f64>>::new(cc, restrictions);

    println!("=== 2D Cubical Sheaf Analysis ===");

    // Test 0-dimensional coboundary (vertices → edges)
    let coboundary_0 = sheaf.coboundary(0);
    println!("\n0-dimensional coboundary (vertices → edges):");
    println!("{coboundary_0}");

    // Expected: 4 edges with varying stalk sizes
    assert_eq!(coboundary_0.nrows(), 10); // Total stalk dimensions
    assert_eq!(coboundary_0.ncols(), 10);
    // Test 1-dimensional coboundary (edges → faces)
    let coboundary_1 = sheaf.coboundary(1);
    println!("\n1-dimensional coboundary (edges → faces):");
    println!("{coboundary_1}");

    // Expected: 1 square with 4×4 stalk, 4 edges with varying stalks
    assert_eq!(coboundary_1.nrows(), 4); // 1 square (4×4), edges total 10
    assert_eq!(coboundary_1.ncols(), 10);

    // Test 2-dimensional coboundary (faces → higher dim, should be empty)
    let coboundary_2 = sheaf.coboundary(2);
    println!("\n2-dimensional coboundary (faces → 3-cubes, should be empty):");
    println!("{coboundary_2}");

    // Should be empty since no 3-cubes
    assert!(coboundary_2.is_empty());
  }
}
