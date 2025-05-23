//! # Complexes Module
//!
//! This module provides data structures and algorithms for working with
//! various types of topological complexes, primarily focusing on cell complexes
//! and simplicial complexes. These are fundamental tools in algebraic topology
//! for representing and analyzing the structure of topological spaces.
//!
//! ## Submodules
//! - [`cell`]: Contains definitions for `Cell` and `CellComplex`, allowing for the construction and
//!   manipulation of regular cell complexes.
//! - [`simplicial`]: Contains definitions for `Simplex` and `SimplicialComplex`, providing tools
//!   for working with simplicial topology, including homology computations.
//! - [`cubical`]: Contains definitions for `Cube` and cubical complex operations.

use std::collections::HashMap;

use super::*;
use crate::{
  definitions::Topology,
  homology::Chain,
  lattice::Lattice,
  set::{Collection, Poset},
};

pub mod cell;
pub mod cubical;
pub mod simplicial;

/// Trait for elements that can be part of a topological complex.
///
/// This trait captures the essential behavior needed for elements (simplices, cubes, etc.)
/// to work with the generic `Complex<T>` structure. Elements must be able to:
/// - Report their dimension
/// - Compute their faces (boundary elements)
/// - Check content equality for deduplication
/// - Handle optional ID assignment when added to a complex
pub trait ComplexElement:
  Clone + std::hash::Hash + Eq + PartialOrd + Ord + std::fmt::Debug {
  /// Returns the dimension of this element.
  fn dimension(&self) -> usize;

  /// Returns all faces (boundary elements) of this element.
  /// For a k-dimensional element, this returns all (k-1)-dimensional faces.
  fn faces(&self) -> Vec<Self>;

  /// Returns the ID if this element has been assigned to a complex, None otherwise.
  fn id(&self) -> Option<usize>;

  /// Checks if this element has the same mathematical content as another.
  fn same_content(&self, other: &Self) -> bool;

  /// Creates a new element with the same content but a specific ID.
  fn with_id(&self, new_id: usize) -> Self;
}

/// A generic topological complex that can work with any type implementing `ComplexElement`.
///
/// This structure uses a lattice to track the face relationships between elements,
/// using their assigned IDs for efficient lattice operations while storing the
/// actual elements separately for easy access.
#[derive(Debug, Clone)]
pub struct Complex<T: ComplexElement> {
  /// The attachment relationships between elements, represented as a lattice of element IDs.
  /// If element `a` is a face of element `b`, then `attachment_lattice.leq(a.id(), b.id())` is
  /// true.
  pub attachment_lattice: Lattice<usize>,

  /// A map storing all elements in the complex, keyed by their assigned ID.
  pub elements: HashMap<usize, T>,

  /// The counter for the next available unique element identifier.
  pub next_id: usize,
}

impl<T: ComplexElement> Complex<T> {
  /// Creates a new, empty complex.
  pub fn new() -> Self {
    Self {
      attachment_lattice: Lattice::new(),
      elements:           HashMap::new(),
      next_id:            0,
    }
  }

  /// Adds an element to the complex along with all its faces.
  ///
  /// This method ensures that the complex satisfies the closure property:
  /// if an element is in the complex, all its faces are also in the complex.
  ///
  /// If an element with equivalent mathematical content already exists, returns the existing
  /// element. Otherwise, assigns a new ID and adds the element and all its faces to the complex.
  pub fn join_element(&mut self, element: T) -> T {
    // Check if we already have this element (by mathematical content)
    if let Some(existing) = self.find_equivalent_element(&element) {
      return existing;
    }

    // Assign a new ID to this element
    let element_with_id =
      if element.id().is_some() && !self.elements.contains_key(&element.id().unwrap()) {
        // Element already has an ID and it's not taken
        if element.id().unwrap() >= self.next_id {
          self.next_id = element.id().unwrap() + 1;
        }
        element
      } else {
        // Assign a new ID
        let new_id = self.next_id;
        self.next_id += 1;
        element.with_id(new_id)
      };

    // Recursively add all faces first
    let mut face_ids = Vec::new();
    for face in element_with_id.faces() {
      let added_face = self.join_element(face);
      face_ids.push(added_face.id().unwrap()); // Safe because we just added it
    }

    // Add the element itself to the lattice
    let element_id = element_with_id.id().unwrap();
    self.attachment_lattice.add_element(element_id);

    // Add face relationships to the lattice
    for face_id in face_ids {
      self.attachment_lattice.add_relation(face_id, element_id);
    }

    // Store the element
    self.elements.insert(element_id, element_with_id.clone());
    element_with_id
  }

  /// Find an existing element with equivalent mathematical content
  fn find_equivalent_element(&self, element: &T) -> Option<T> {
    self.elements.values().find(|existing| element.same_content(existing)).cloned()
  }

  /// Returns the element with the given ID, if it exists.
  pub fn get_element(&self, id: usize) -> Option<&T> { self.elements.get(&id) }

  /// Returns all elements of a specific dimension.
  pub fn elements_of_dimension(&self, dimension: usize) -> Vec<T> {
    self.elements.values().filter(|element| element.dimension() == dimension).cloned().collect()
  }

  /// Returns the maximum dimension of any element in the complex.
  pub fn max_dimension(&self) -> usize {
    self.elements.values().map(|element| element.dimension()).max().unwrap_or(0)
  }

  /// Returns all direct faces of the given element.
  pub fn faces(&self, element: &T) -> Vec<T> {
    if let Some(id) = element.id() {
      self
        .attachment_lattice
        .predecessors(id)
        .into_iter()
        .filter_map(|face_id| self.get_element(face_id).cloned())
        .collect()
    } else {
      Vec::new()
    }
  }

  /// Returns all direct cofaces (attachments) of the given element.
  pub fn cofaces(&self, element: &T) -> Vec<T> {
    if let Some(id) = element.id() {
      self
        .attachment_lattice
        .successors(id)
        .into_iter()
        .filter_map(|coface_id| self.get_element(coface_id).cloned())
        .collect()
    } else {
      Vec::new()
    }
  }

  pub fn homology<R: Ring + Copy>(&self, dimension: usize) -> Chain<Self, R> { todo!() }
}

impl<T: ComplexElement> Default for Complex<T> {
  fn default() -> Self { Self::new() }
}

impl<T: ComplexElement> Collection for Complex<T> {
  type Item = T;

  fn contains(&self, point: &Self::Item) -> bool {
    if let Some(id) = point.id() {
      self.elements.contains_key(&id)
    } else {
      false
    }
  }

  fn is_empty(&self) -> bool { self.elements.is_empty() }
}

impl<T: ComplexElement> Poset for Complex<T> {
  fn leq(&self, a: &Self::Item, b: &Self::Item) -> Option<bool> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self.attachment_lattice.leq(&id_a, &id_b),
      _ => None,
    }
  }

  fn upset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .upset(id)
        .into_iter()
        .filter_map(|upset_id| self.get_element(upset_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn downset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .downset(id)
        .into_iter()
        .filter_map(|downset_id| self.get_element(downset_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn minimal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .minimal_elements()
      .into_iter()
      .filter_map(|id| self.get_element(id).cloned())
      .collect()
  }

  fn maximal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .maximal_elements()
      .into_iter()
      .filter_map(|id| self.get_element(id).cloned())
      .collect()
  }

  fn join(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self
        .attachment_lattice
        .join(id_a, id_b)
        .and_then(|join_id| self.get_element(join_id).cloned()),
      _ => None,
    }
  }

  fn meet(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self
        .attachment_lattice
        .meet(id_a, id_b)
        .and_then(|meet_id| self.get_element(meet_id).cloned()),
      _ => None,
    }
  }

  fn successors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .successors(id)
        .into_iter()
        .filter_map(|succ_id| self.get_element(succ_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn predecessors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .predecessors(id)
        .into_iter()
        .filter_map(|pred_id| self.get_element(pred_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }
}

impl<T: ComplexElement> Topology for Complex<T> {
  fn neighborhood(&self, item: &Self::Item) -> Vec<Self::Item> {
    // Return direct cofaces (elements that have this item as a face)
    self.cofaces(item)
  }

  fn boundary<R: Ring + Copy>(&self, item: &Self::Item) -> Chain<Self, R> {
    if item.dimension() == 0 {
      return Chain::new(self);
    }

    let mut boundary_chain_items = Vec::new();
    let mut boundary_chain_coeffs = Vec::new();

    // Get faces from the element itself (which computes them mathematically)
    let faces = item.faces();

    for (i, face) in faces.into_iter().enumerate() {
      // Use alternating signs for the boundary operator
      let coeff = if i % 2 == 0 { R::one() } else { -R::one() };
      boundary_chain_items.push(face);
      boundary_chain_coeffs.push(coeff);
    }

    Chain::from_items_and_coeffs(self, boundary_chain_items, boundary_chain_coeffs)
  }
}

// Type aliases for convenience
pub use cubical::Cube;
pub use simplicial::Simplex;

/// A simplicial complex using the generic Complex structure
pub type SimplicialComplex = Complex<Simplex>;

/// A cubical complex using the generic Complex structure  
pub type CubicalComplex = Complex<Cube>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_generic_complex_with_simplex() {
    let mut complex: Complex<Simplex> = Complex::new();

    // Create a triangle (2-simplex)
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    // Should have 1 triangle, 3 edges, 3 vertices
    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert_eq!(complex.elements_of_dimension(1).len(), 3);
    assert_eq!(complex.elements_of_dimension(0).len(), 3);

    // Check that the triangle is in the complex (use the returned element with ID)
    assert!(complex.contains(&added_triangle));

    // Check lattice relationships
    let faces = complex.faces(&added_triangle);
    assert_eq!(faces.len(), 3); // Triangle should have 3 edges as direct faces
  }

  #[test]
  fn test_generic_complex_with_cube() {
    let mut complex: Complex<Cube> = Complex::new();

    // Create a square (2-cube)
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);

    // Should have 1 square, 4 edges, 4 vertices
    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert_eq!(complex.elements_of_dimension(1).len(), 4);
    assert_eq!(complex.elements_of_dimension(0).len(), 4);

    // Check that the square is in the complex (use the returned element with ID)
    assert!(complex.contains(&added_square));

    // Check lattice relationships
    let faces = complex.faces(&added_square);
    assert_eq!(faces.len(), 4); // Square should have 4 edges as direct faces
  }

  #[test]
  fn test_simplicial_complex_type_alias() {
    let mut complex = SimplicialComplex::new();

    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert!(complex.contains(&added_triangle));

    // Test Poset operations
    let faces = complex.faces(&added_triangle);
    assert_eq!(faces.len(), 3);

    // Test that all faces are actually contained in the complex
    for face in &faces {
      assert!(complex.contains(face));
    }
  }

  #[test]
  fn test_cubical_complex_type_alias() {
    let mut complex = CubicalComplex::new();

    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);

    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert!(complex.contains(&added_square));

    // Test Poset operations
    let faces = complex.faces(&added_square);
    assert_eq!(faces.len(), 4);

    // Test that all faces are actually contained in the complex
    for face in &faces {
      assert!(complex.contains(face));
    }
  }

  #[test]
  fn test_complex_poset_operations() {
    let mut complex = SimplicialComplex::new();

    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    // Get the actual elements with IDs from the complex
    let vertices = complex.elements_of_dimension(0);
    let edges = complex.elements_of_dimension(1);

    // Find specific vertex and edge by content
    let vertex = vertices.iter().find(|v| v.vertices() == &[0]).unwrap().clone();
    let edge = edges.iter().find(|e| e.vertices() == &[0, 1]).unwrap().clone();

    // Test leq relationships
    assert_eq!(complex.leq(&vertex, &edge), Some(true));
    assert_eq!(complex.leq(&edge, &added_triangle), Some(true));
    assert_eq!(complex.leq(&vertex, &added_triangle), Some(true));
    assert_eq!(complex.leq(&added_triangle, &vertex), Some(false));

    // Test upset/downset
    let vertex_upset = complex.upset(vertex.clone());
    assert!(vertex_upset.contains(&vertex));
    assert!(vertex_upset.contains(&edge));
    assert!(vertex_upset.contains(&added_triangle));

    let triangle_downset = complex.downset(added_triangle.clone());
    assert!(triangle_downset.contains(&vertex));
    assert!(triangle_downset.contains(&edge));
    assert!(triangle_downset.contains(&added_triangle));
  }

  #[test]
  fn test_complex_topology_operations() {
    let mut complex = SimplicialComplex::new();

    let edge = Simplex::new(1, vec![0, 1]);
    let added_edge = complex.join_element(edge);

    // Get the actual vertex with ID from the complex
    let vertices = complex.elements_of_dimension(0);
    let vertex = vertices.iter().find(|v| v.vertices() == &[0]).unwrap().clone();

    // Test neighborhood (cofaces)
    let vertex_neighborhood = complex.neighborhood(&vertex);
    assert_eq!(vertex_neighborhood.len(), 1);
    assert!(vertex_neighborhood.contains(&added_edge));

    let edge_neighborhood = complex.neighborhood(&added_edge);
    assert_eq!(edge_neighborhood.len(), 0); // No 2-simplices attached
  }

  #[test]
  fn test_mixed_complex_operations() {
    // Test that we can use the same generic operations on both simplicial and cubical complexes

    // Simplicial complex
    let mut simplicial_complex = SimplicialComplex::new();
    let triangle = Simplex::from_vertices(vec![0, 1, 2]);
    let added_triangle = simplicial_complex.join_element(triangle);

    // Should automatically add all faces
    assert_eq!(simplicial_complex.elements_of_dimension(2).len(), 1);
    assert_eq!(simplicial_complex.elements_of_dimension(1).len(), 3);
    assert_eq!(simplicial_complex.elements_of_dimension(0).len(), 3);

    // Cubical complex
    let mut cubical_complex = CubicalComplex::new();
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = cubical_complex.join_element(square);

    // Should automatically add all faces
    assert_eq!(cubical_complex.elements_of_dimension(2).len(), 1);
    assert_eq!(cubical_complex.elements_of_dimension(1).len(), 4);
    assert_eq!(cubical_complex.elements_of_dimension(0).len(), 4);

    // Both should support the same interface
    assert!(simplicial_complex.contains(&added_triangle));
    assert!(cubical_complex.contains(&added_square));

    // Both should have proper lattice structure
    let triangle_faces = simplicial_complex.faces(&added_triangle);
    let square_faces = cubical_complex.faces(&added_square);
    assert_eq!(triangle_faces.len(), 3); // triangle has 3 edges
    assert_eq!(square_faces.len(), 4); // square has 4 edges
  }

  #[test]
  fn test_automatic_id_assignment() {
    let mut complex = SimplicialComplex::new();

    // Create elements
    let vertex1 = Simplex::new(0, vec![0]);
    let vertex2 = Simplex::new(0, vec![1]);
    let edge = Simplex::new(1, vec![0, 1]);

    // Add them to the complex - should all be properly stored
    let added_vertex1 = complex.join_element(vertex1);
    let added_vertex2 = complex.join_element(vertex2);
    let added_edge = complex.join_element(edge);

    // All should be contained
    assert!(complex.contains(&added_vertex1));
    assert!(complex.contains(&added_vertex2));
    assert!(complex.contains(&added_edge));

    // Should have proper lattice relationships
    assert!(complex.leq(&added_vertex1, &added_edge).unwrap());
    assert!(complex.leq(&added_vertex2, &added_edge).unwrap());
  }
}
