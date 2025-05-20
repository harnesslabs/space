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
use std::{collections::HashMap, hash::Hash, marker::PhantomData};

use harness_algebra::tensors::dynamic::{
  matrix::{DynamicDenseMatrix, RowMajor},
  vector::DynamicVector,
};

use crate::{definitions::Topology, set::Poset};

pub struct Sheaf<T: Topology + Poset, C: Category>
where T::Item: Hash + Eq {
  space:        T,
  restrictions: HashMap<(T::Item, T::Item), C::Morphism>,
}

impl<T: Topology + Poset, C: Category> Sheaf<T, C>
where
  T::Item: Hash + Eq + Clone,
  C: Clone,
{
  pub fn new(space: T) -> Self { Self { space, restrictions: HashMap::new() } }

  pub fn restrict(&self, item: T::Item, restriction: T::Item, section: HashMap<T::Item, C>) -> C {
    assert!(self.space.leq(&item, &restriction).unwrap());
    let restriction = self.restrictions.get(&(item.clone(), restriction)).unwrap();
    let data = section.get(&item).unwrap().clone();
    restriction(data)
  }
}

pub trait Category: Sized {
  type Morphism: Fn(Self) -> Self;

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism;
  fn identity(a: Self) -> Self::Morphism;
}

impl<T> Category for DynamicVector<T> {
  type Morphism = DynamicDenseMatrix<T, RowMajor>;

  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism { Box::new(move |x| f(g(x))) }

  fn identity(a: Self) -> Self::Morphism { todo!() }
}

// // /// A trait representing a cellular sheaf over a topological space.
// // ///
// // /// A sheaf assigns a vector space (stalk) to each cell and provides
// // /// restriction maps between stalks when cells are incident.
// // pub trait Presheaf<T: TopologicalSpace> {
// //   /// The type of the data at each point
// //   type Data;
// //   /// The type of sections over open sets
// //   type Section: Section<T, Stalk = Self::Data> + Clone;

// //   /// Restricts a section from a larger open set to a smaller one
// //   fn restrict(
// //     &self,
// //     section: &Self::Section,
// //     from: &<T as TopologicalSpace>::OpenSet,
// //     to: &<T as TopologicalSpace>::OpenSet,
// //   ) -> Self::Section;
// // }

// /// A trait representing a sheaf over a topological space, extending `Presheaf`.
// ///
// /// A sheaf satisfies the gluing axiom: locally compatible sections can be uniquely glued
// /// to a global section over the union of their domains.
// pub trait Sheaf<T: TopologicalSpace>: Presheaf<T>
// where <T as TopologicalSpace>::OpenSet: Clone {
//   /// Attempts to glue a list of local sections into a single global section.
//   ///
//   /// Returns `Some(section)` if all sections agree on overlaps, giving a section over
//   /// the union of their domains; otherwise returns `None`
//   fn glue(&self, sections: &[Self::Section]) -> Option<Self::Section>
//   where <Self as Presheaf<T>>::Section: 'static {
//     // collect domains
//     let domains: Vec<_> = sections.iter().map(Self::Section::domain).collect();
//     let sections = sections.to_vec();
//     // union them all up
//     let mut big_union = domains.first().cloned()?;
//     for dom in domains.iter().skip(1) {
//       big_union = big_union.join(dom);
//     }

//     // check pairwise compatibility
//     for (i, si) in sections.iter().enumerate() {
//       let ui = si.domain();
//       for sj in sections.iter().skip(i + 1) {
//         let uj = sj.domain();
//         let overlap = ui.meet(&uj);
//         if !overlap.is_empty() {
//           let ri = self.restrict(si, &ui, &overlap);
//           let rj = self.restrict(sj, &uj, &overlap);
//           if ri != rj {
//             // conflict on the overlap → no glue
//             return None;
//           }
//         }
//       }
//     }

//     // piecewise construction
//     //—you'll need a constructor like Section::from_closure(domain, f)
//     Some(Self::Section::from_closure(big_union, move |pt| {
//       // pick the first local section whose domain contains pt
//       for sec in sections.clone() {
//         if sec.domain().contains(pt) {
//           return sec.evaluate(pt);
//         }
//       }
//       // outside all domains? should never happen since pt∈big_union
//       None
//     }))
//   }
// }

// /// A trait representing a section of a presheaf over an open set.
// ///
// /// A section assigns to each point in its domain an element of the stalk,
// /// and two sections can be compared for equality on overlaps.
// pub trait Section<T: TopologicalSpace>: PartialEq {
//   /// The type of the stalk this section takes values in
//   type Stalk;

//   /// Evaluates the section at a point in its domain, returning an element of the stalk
//   fn evaluate(&self, point: &<T as TopologicalSpace>::Point) -> Option<Self::Stalk>;

//   /// Gets the open set over which this section is defined
//   fn domain(&self) -> <T as TopologicalSpace>::OpenSet;

//   /// Construct a section by giving its domain and a pointwise evaluation function
//   fn from_closure<F>(domain: <T as TopologicalSpace>::OpenSet, f: F) -> Self
//   where F: Fn(&<T as TopologicalSpace>::Point) -> Option<Self::Stalk>;
// }

// #[cfg(test)]
// mod tests {
//   use std::collections::HashSet;

//   use super::*;

//   fn create_test_graph() -> Graph<usize, Undirected> {
//     let mut vertices = HashSet::new();
//     vertices.insert(1);
//     vertices.insert(2);
//     vertices.insert(3);

//     let mut edges = HashSet::new();
//     edges.insert((1, 2));
//     edges.insert((2, 3));

//     Graph::new(vertices, edges)
//   }

//   // Helper to create a test GraphSheaf with restriction matrices
//   fn create_test_sheaf() -> GraphSheaf<f64, usize> {
//     let graph = create_test_graph();
//     let vertex_dim = 2;
//     let edge_dim = 1;

//     let mut sheaf = GraphSheaf::new(graph, vertex_dim, edge_dim);

//     // Add restriction matrices for each edge
//     // For vertex 1 to edge (1,2)
//     let r1_12 = vec![vec![1.0], vec![0.0]]; // Projects to first component
//     sheaf.restriction_matrices.insert((1, 2), r1_12);

//     // For vertex 2 to edge (1,2)
//     let r2_12 = vec![vec![0.0], vec![1.0]]; // Projects to second component
//     sheaf.restriction_matrices.insert((2, 1), r2_12);

//     // For vertex 2 to edge (2,3)
//     let r2_23 = vec![vec![1.0], vec![0.0]]; // Projects to first component
//     sheaf.restriction_matrices.insert((2, 3), r2_23);

//     // For vertex 3 to edge (2,3)
//     let r3_23 = vec![vec![0.0], vec![1.0]]; // Projects to second component
//     sheaf.restriction_matrices.insert((3, 2), r3_23);

//     sheaf
//   }

//   // Helper to create DynVector from a slice of values
//   fn create_vector(values: &[f64]) -> DynamicVector<f64> { DynamicVector::from(values) }

//   #[test]
//   fn test_restriction_single_point() {
//     let sheaf = create_test_sheaf();

//     // Create a section over a single point
//     let mut section = HashMap::new();
//     section.insert(VertexOrEdge::Vertex(1), create_vector(&[3.0, 4.0]));

//     // Create the from and to sets
//     let from = {
//       let mut set = HashSet::new();
//       set.insert(VertexOrEdge::Vertex(1));
//       set
//     };

//     let to = from.clone(); // Same set for this test

//     // Restrict the section
//     let restricted = sheaf.restrict(&section, &from, &to);

//     // The restricted section should be identical
//     assert_eq!(restricted.len(), 1);
//     assert_eq!(restricted[&VertexOrEdge::Vertex(1)], create_vector(&[3.0, 4.0]));
//   }

//   #[test]
//   fn test_restriction_subset() {
//     let sheaf = create_test_sheaf();

//     // Create a section over multiple points
//     let mut section = HashMap::new();
//     section.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));
//     section.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));
//     section.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Create the from and to sets
//     let from = {
//       let mut set = HashSet::new();
//       set.insert(VertexOrEdge::Vertex(1));
//       set.insert(VertexOrEdge::Vertex(2));
//       set.insert(VertexOrEdge::Vertex(3));
//       set
//     };

//     let to = {
//       let mut set = HashSet::new();
//       set.insert(VertexOrEdge::Vertex(1));
//       set.insert(VertexOrEdge::Vertex(3));
//       set
//     };

//     // Restrict the section
//     let restricted = sheaf.restrict(&section, &from, &to);

//     // The restricted section should only have points from 'to'
//     assert_eq!(restricted.len(), 2);
//     assert_eq!(restricted[&VertexOrEdge::Vertex(1)], create_vector(&[1.0, 2.0]));
//     assert_eq!(restricted[&VertexOrEdge::Vertex(3)], create_vector(&[5.0, 6.0]));
//     assert!(!restricted.contains_key(&VertexOrEdge::Vertex(2)));
//   }

//   #[test]
//   fn test_glue_compatible_sections() {
//     let sheaf = create_test_sheaf();

//     // Create two compatible sections with overlap
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));
//     section1.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0])); // Same as in section1
// for overlap     section2.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2]);

//     // The gluing should succeed
//     assert!(glued.is_some());
//     let glued_section = glued.unwrap();

//     // The glued section should contain all three vertices
//     assert_eq!(glued_section.len(), 3);
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(1)], create_vector(&[1.0, 2.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(2)], create_vector(&[3.0, 4.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(3)], create_vector(&[5.0, 6.0]));
//   }

//   #[test]
//   fn test_glue_incompatible_sections() {
//     let sheaf = create_test_sheaf();

//     // Create two incompatible sections with a disagreement at the overlap
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));
//     section1.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Vertex(2), create_vector(&[3.5, 4.5])); // Different from
// section1     section2.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2]);

//     // The gluing should fail due to incompatibility
//     assert!(glued.is_none());
//   }

//   #[test]
//   fn test_glue_disjoint_sections() {
//     let sheaf = create_test_sheaf();

//     // Create two disjoint sections (no overlap)
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2]);

//     // The gluing should succeed since there's no overlap to check
//     assert!(glued.is_some());
//     let glued_section = glued.unwrap();

//     // The glued section should contain both vertices
//     assert_eq!(glued_section.len(), 2);
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(1)], create_vector(&[1.0, 2.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(3)], create_vector(&[5.0, 6.0]));
//   }

//   #[test]
//   fn test_glue_three_sections() {
//     let sheaf = create_test_sheaf();

//     // Create three sections with various overlaps
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));

//     let mut section3 = HashMap::new();
//     section3.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2, section3]);

//     // The gluing should succeed
//     assert!(glued.is_some());
//     let glued_section = glued.unwrap();

//     // The glued section should contain all three vertices
//     assert_eq!(glued_section.len(), 3);
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(1)], create_vector(&[1.0, 2.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(2)], create_vector(&[3.0, 4.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(3)], create_vector(&[5.0, 6.0]));
//   }

//   #[test]
//   fn test_glue_three_incompatible_sections() {
//     let sheaf = create_test_sheaf();

//     // Create three sections where two are compatible but one is not
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));
//     section1.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0])); // Same as section1
//     section2.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     let mut section3 = HashMap::new();
//     section3.insert(VertexOrEdge::Vertex(2), create_vector(&[3.5, 4.5])); // Different at vertex
// 2     section3.insert(VertexOrEdge::Vertex(3), create_vector(&[5.0, 6.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2, section3]);

//     // The gluing should fail due to incompatibility
//     assert!(glued.is_none());
//   }

//   #[test]
//   fn test_glue_with_edge_points() {
//     let sheaf = create_test_sheaf();

//     // Create sections that include both vertex and edge points
//     let mut section1 = HashMap::new();
//     section1.insert(VertexOrEdge::Vertex(1), create_vector(&[1.0, 2.0]));
//     section1.insert(VertexOrEdge::Edge(1, 2), create_vector(&[1.0])); // Edge point

//     let mut section2 = HashMap::new();
//     section2.insert(VertexOrEdge::Edge(1, 2), create_vector(&[1.0])); // Same as in section1
//     section2.insert(VertexOrEdge::Vertex(2), create_vector(&[3.0, 4.0]));

//     // Attempt to glue the sections
//     let glued = sheaf.glue(&[section1, section2]);

//     // The gluing should succeed
//     assert!(glued.is_some());
//     let glued_section = glued.unwrap();

//     // The glued section should contain all points
//     assert_eq!(glued_section.len(), 3);
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(1)], create_vector(&[1.0, 2.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Edge(1, 2)], create_vector(&[1.0]));
//     assert_eq!(glued_section[&VertexOrEdge::Vertex(2)], create_vector(&[3.0, 4.0]));
//   }

//   #[test]
//   fn test_empty_glue() {
//     let sheaf = create_test_sheaf();

//     // Attempt to glue an empty list of sections
//     let sections: Vec<HashMap<VertexOrEdge<usize>, DynamicVector<f64>>> = vec![];
//     let glued = sheaf.glue(&sections);

//     // The gluing should fail (can't glue nothing)
//     assert!(glued.is_none());
//   }
// }
