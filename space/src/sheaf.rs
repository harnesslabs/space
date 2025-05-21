//! # Sheaves on Graphs
//!
//! This module provides an implementation of cellular sheaves over topological spaces,
//! with a specific focus on undirected graphs.
//!
//! ## Concepts
//!
//! A sheaf is a mathematical structure that assigns data (vector spaces in this case) to
//! points in a topological space, along with rules for how this data restricts between
//! related points. Key components include:
//!
//! - **Stalks**: Vector spaces assigned to each point (vertices and edges in a graph)
//! - **Restriction Maps**: Linear transformations between stalks of related points
//! - **Sections**: Assignments of stalk elements to points that respect restriction maps
//!
//! ## Core Traits
//!
//! - `Presheaf`: Defines the basic structure of assigning data to points with restriction maps
//! - `Sheaf`: Extends `Presheaf` with the ability to glue compatible local sections into global
//!   ones
//! - `Section`: Represents an assignment of data over an open set that can be evaluated at points
//!
//! ## Implementation
//!
//! `GraphSheaf` implements cellular sheaves over undirected graphs where:
//!
//! - Vertices and edges have stalks of possibly different dimensions
//! - Restriction maps are specified as matrices
//! - Sections are stored as hashmaps from graph points to vector values
//!
//! The implementation supports:
//! - Restricting sections from larger to smaller domains
//! - Gluing compatible sections (ones that agree on overlaps) into a single global section
//!
//! This structure is fundamental in applications like distributed consensus, signal processing
//! on graphs, and modeling systems where local data must satisfy global constraints.
use std::{collections::HashMap, fmt::Debug, hash::Hash};

use harness_algebra::{
  rings::Field,
  tensors::dynamic::{
    matrix::{DynamicDenseMatrix, RowMajor},
    vector::DynamicVector,
  },
};

use crate::{definitions::Topology, set::Poset};

// TODO: We should make this have a nice construction setup so you can build the underlying space
// and the restrictions simultaneously
pub struct Sheaf<T: Topology + Poset, C: Category>
where T::Item: Hash + Eq {
  space:        T,
  restrictions: HashMap<(T::Item, T::Item), C::Morphism>,
}

impl<T, C> Sheaf<T, C>
where
  T: Topology + Poset,
  T::Item: Hash + Eq + Clone + Debug,
  C: Category + Clone + Eq + Debug,
  C::Morphism: Clone + Debug,
{
  pub fn new(space: T, restrictions: HashMap<(T::Item, T::Item), C::Morphism>) -> Self {
    assert!(restrictions.iter().all(|(k, v)| space.leq(&k.0, &k.1).unwrap()));
    // TODO: Assert that there is a restriction for every "upset" of points
    Self { space, restrictions }
  }

  pub fn restrict(&self, item: T::Item, restriction: T::Item, section: HashMap<T::Item, C>) -> C {
    assert!(self.space.leq(&item, &restriction).unwrap());
    let restriction = self.restrictions.get(&(item.clone(), restriction)).unwrap();
    let data = section.get(&item).unwrap().clone();
    C::apply(restriction.clone(), data)
  }

  pub fn is_global_section(&self, section: HashMap<T::Item, C>) -> bool {
    // TODO: Go through the poset and check if the section is compatible with the restrictions
    let space = &self.space;
    for element in self.space.minimal_elements() {
      dbg!(&element);
      let upset = space.upset(element.clone());
      for other in upset {
        dbg!(&other);
        let restriction = self.restrictions.get(&(element.clone(), other.clone())).unwrap();
        dbg!(&restriction);
        let data = section.get(&element).unwrap().clone();
        dbg!(&data);
        let restricted = C::apply(restriction.clone(), data);
        dbg!(&restricted);
        if !section.contains_key(&element) {
          return false;
        }
        let data = section.get(&other).unwrap().clone();
        if !(data == restricted) {
          return false;
        }
      }
    }
    true
  }
}

pub trait Category: Sized {
  type Morphism;

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism;
  fn identity(a: Self) -> Self::Morphism;
  fn apply(f: Self::Morphism, x: Self) -> Self;
}

impl<F: Field + Copy> Category for DynamicVector<F> {
  type Morphism = DynamicDenseMatrix<F, RowMajor>;

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism { todo!() }

  fn identity(a: Self) -> Self::Morphism { todo!() }

  fn apply(f: Self::Morphism, x: Self) -> Self { f * x }
}

#[cfg(test)]
mod tests {
  use harness_algebra::{modular, prime_field};

  use super::*;
  use crate::complexes::cell::{Cell, CellComplex};

  modular!(Mod7, u32, 7);
  prime_field!(Mod7);

  fn cell_complex_1d(
  ) -> (CellComplex, HashMap<(Cell, Cell), DynamicDenseMatrix<Mod7, RowMajor>>, Cell, Cell, Cell)
  {
    let mut cc = CellComplex::new();
    let v1 = cc.add_cell(0, vec![]);
    let v2 = cc.add_cell(0, vec![]);
    let e1 = cc.add_cell(1, vec![&v1, &v2]);
    let restrictions = HashMap::from([
      ((v1.clone(), e1.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((v2.clone(), e1.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
    ]);
    (cc, restrictions, v1, v2, e1)
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
    let v1 = cc.add_cell(0, vec![]); // R^1
    let v2 = cc.add_cell(0, vec![]); // R^2
    let v3 = cc.add_cell(0, vec![]); // R^3
                                     // Edges
    let e12 = cc.add_cell(1, vec![&v1, &v2]); // R^2
    let e13 = cc.add_cell(1, vec![&v2, &v3]); // R^2
    let e23 = cc.add_cell(1, vec![&v3, &v1]); // R^2
                                              // Faces
    let f123 = cc.add_cell(2, vec![&e12, &e13, &e23]); // R^3

    let restrictions = HashMap::from([
      ((v1.clone(), e12.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((v2.clone(), e12.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v1.clone(), e13.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v3.clone(), e13.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v2.clone(), e23.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v3.clone(), e23.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((e12.clone(), f123.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((e13.clone(), f123.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((e23.clone(), f123.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
    ]);
    (cc, restrictions, v1, v2, v3, e12, e13, e23, f123)
  }

  #[test]
  fn test_sheaf_global_section_1d() {
    let (cc, restrictions, v1, v2, e1) = cell_complex_1d();

    let sheaf = Sheaf::<CellComplex, DynamicVector<Mod7>>::new(cc, restrictions);

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(sheaf.is_global_section(section));

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(section));

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(section));

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(3), Mod7::from(3)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(section));
  }
}
