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

use super::*;
use crate::{definitions::Topology, set::Poset};

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
  C: Category + Clone + Eq + Debug,
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

#[cfg(test)]
mod tests {
  #![allow(clippy::type_complexity)]
  use harness_algebra::{
    modular, prime_field,
    tensors::dynamic::{
      matrix::{DynamicDenseMatrix, RowMajor},
      vector::DynamicVector,
    },
  };

  use super::*;
  use crate::complexes::cell::{Cell, CellComplex};

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);

  fn cell_complex_1d(
  ) -> (CellComplex, HashMap<(Cell, Cell), DynamicDenseMatrix<Mod7, RowMajor>>, Cell, Cell, Cell)
  {
    let mut cc = CellComplex::new();
    let v1 = cc.add_cell(0, &[]);
    let v2 = cc.add_cell(0, &[]);
    let e1 = cc.add_cell(1, &[&v1, &v2]);
    let restrictions = HashMap::from([
      ((v1, e1), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((v2, e1), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
    ]);
    (cc, restrictions, v1, v2, e1)
  }

  #[test]
  fn test_sheaf_global_section_1d() {
    let (cc, restrictions, v1, v2, e1) = cell_complex_1d();

    let sheaf = Sheaf::<CellComplex, DynamicVector<Mod7>>::new(cc, restrictions);

    let section = HashMap::from([
      (v1, DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1, DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1, DynamicVector::<Mod7>::new(vec![Mod7::from(1)])), // R^1
      (v2, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1, DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1, DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1, DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2, DynamicVector::<Mod7>::new(vec![Mod7::from(3), Mod7::from(3)])), // R^2
      (e1, DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));
  }

  fn cell_complex_2d() -> (
    CellComplex,
    HashMap<(Cell, Cell), DynamicDenseMatrix<Mod7, RowMajor>>,
    Cell,
    Cell,
    Cell,
    Cell,
    Cell,
    Cell,
    Cell,
  ) {
    let mut cc = CellComplex::new();
    // Vertices
    let v0 = cc.add_cell(0, &[]); // R^1
    let v1 = cc.add_cell(0, &[]); // R^2
    let v2 = cc.add_cell(0, &[]); // R^3
                                  // Edges
    let e01 = cc.add_cell(1, &[&v0, &v1]); // R^2
    let e02 = cc.add_cell(1, &[&v0, &v2]); // R^2
    let e12 = cc.add_cell(1, &[&v1, &v2]); // R^2

    // Faces
    let f012 = cc.add_cell(2, &[&e01, &e02, &e12]); // R^3

    let restrictions = HashMap::from([
      ((v0, e01), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((v1, e01), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(1)]));
        mat
      }),
      ((v0, e02), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat
      }),
      ((v2, e02), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat
      }),
      ((v1, e12), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v2, e12), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat
      }),
      ((e01, f012), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(2),
          Mod7::from(0),
          Mod7::from(0),
        ]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(0),
          Mod7::from(0),
          Mod7::from(0),
        ]));
        mat
      }),
      ((e02, f012), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(2),
          Mod7::from(0),
          Mod7::from(0),
        ]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(0),
          Mod7::from(1),
          Mod7::from(0),
        ]));
        mat
      }),
      ((e12, f012), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(1),
          Mod7::from(0),
          Mod7::from(0),
        ]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![
          Mod7::from(0),
          Mod7::from(0),
          Mod7::from(0),
        ]));
        mat
      }),
    ]);

    (cc, restrictions, v0, v1, v2, e01, e02, e12, f012)
  }

  #[test]
  fn test_sheaf_global_section_2d() {
    let (cc, restrictions, v0, v1, v2, e01, e02, e12, f012) = cell_complex_2d();

    let sheaf = Sheaf::<CellComplex, DynamicVector<Mod7>>::new(cc, restrictions);

    let section = HashMap::from([
      (v0, DynamicVector::<Mod7>::new(vec![Mod7::from(1)])), // R^1
      (v1, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (v2, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2), Mod7::from(3)])), // R^3
      (e01, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e02, DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)])), // R^2
      (e12, DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
      (f012, DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0), Mod7::from(0)])), // R^3
    ]);
    assert!(sheaf.is_global_section(&section));
  }
}
