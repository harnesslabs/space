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
  pub space:        T,
  pub restrictions: HashMap<(T::Item, T::Item), C::Morphism>,
}

impl<T, C> Sheaf<T, C>
where
  T: Topology + Poset,
  T::Item: Hash + Eq + Clone + Debug,
  C: Category + Clone + Eq + Debug,
  C::Morphism: Clone + Debug,
{
  /// TODO: We should make a builder API of some kind that forces the restrictions to be defined for
  /// all the successors at time of creation. In an ideal world, there would also be checking that
  /// the dimensions of matrices (for example) work out.
  pub fn new(space: T, restrictions: HashMap<(T::Item, T::Item), C::Morphism>) -> Self {
    assert!(restrictions.iter().all(|(k, v)| space.leq(&k.0, &k.1).unwrap()));

    Self { space, restrictions }
  }

  pub fn restrict(&self, item: T::Item, restriction: T::Item, section: HashMap<T::Item, C>) -> C {
    assert!(self.space.leq(&item, &restriction).unwrap());
    let restriction = self.restrictions.get(&(item.clone(), restriction)).unwrap();
    let data = section.get(&item).unwrap().clone();
    C::apply(restriction.clone(), data)
  }

  pub fn is_global_section(&self, section: &HashMap<T::Item, C>) -> bool {
    for ((parent_item, child_item), restriction_map) in &self.restrictions {
      let Some(parent_data) = section.get(parent_item) else {
        return false;
      };

      let Some(child_data_from_section) = section.get(child_item) else {
        return false;
      };

      let restricted_parent_data = C::apply(restriction_map.clone(), parent_data.clone());

      if restricted_parent_data != *child_data_from_section {
        return false;
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

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism { f * g }

  fn identity(a: Self) -> Self::Morphism {
    let mut mat = DynamicDenseMatrix::<F, RowMajor>::new();
    for i in 0..a.dimension() {
      let mut col = Self::from(vec![F::zero(); a.dimension()]);
      col.components[i] = F::one();
      mat.append_column(&col);
    }
    mat
  }

  fn apply(f: Self::Morphism, x: Self) -> Self { f * x }
}

#[cfg(test)]
mod tests {
  #![allow(clippy::type_complexity)]
  use harness_algebra::{modular, prime_field};

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

  #[test]
  fn test_sheaf_global_section_1d() {
    let (cc, restrictions, v1, v2, e1) = cell_complex_1d();

    let sheaf = Sheaf::<CellComplex, DynamicVector<Mod7>>::new(cc, restrictions);

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(4)])), // R^2
    ]);
    assert!(!sheaf.is_global_section(&section));

    let section = HashMap::from([
      (v1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(2)])), // R^1
      (v2.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
      (e1.clone(), DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)])), // R^2
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
      ((v0.clone(), e01.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(2)]));
        mat
      }),
      ((v1.clone(), e01.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(1)]));
        mat
      }),
      ((v0.clone(), e02.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat
      }),
      ((v2.clone(), e02.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(1), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat
      }),
      ((v1.clone(), e12.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat
      }),
      ((v2.clone(), e12.clone()), {
        let mut mat = DynamicDenseMatrix::<Mod7, RowMajor>::new();
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(2), Mod7::from(0)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(2)]));
        mat.append_column(&DynamicVector::<Mod7>::new(vec![Mod7::from(0), Mod7::from(0)]));
        mat
      }),
      ((e01.clone(), f012.clone()), {
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
      ((e02.clone(), f012.clone()), {
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
      ((e12.clone(), f012.clone()), {
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
