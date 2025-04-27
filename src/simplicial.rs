use std::array;

use itertools::Itertools;

pub struct Simplex<V> {
  // A simplex of dimension K has K+1 vertices
  vertices: Vec<V>,
  dimension: usize,
}

impl<V: Clone + PartialEq> Simplex<V> {
  // Create a new simplex from K+1 vertices
  pub fn new(dimension: usize, vertices: Vec<V>) -> Self {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices, dimension }
  }

  // Get a reference to the vertices
  pub fn vertices(&self) -> &[V] {
    &self.vertices
  }

  pub fn faces(&self) -> Vec<Simplex<V>> {
    self
      .vertices
      .clone()
      .into_iter()
      .combinations(self.dimension)
      .map(|v| Self::new(self.dimension - 1, v))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_simplex_faces() {
    let simplex = Simplex::new(2, vec![0, 1, 2]);
    let faces = simplex.faces();
    assert_eq!(faces.len(), 3);
    assert_eq!(faces[0].vertices(), &[0, 1]);
    assert_eq!(faces[1].vertices(), &[0, 2]);
    assert_eq!(faces[2].vertices(), &[1, 2]);
  }
}
