use std::ops::{Add, Mul};

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct Simplex<V> {
  vertices: Vec<V>,
  dimension: usize,
}

impl<V: Clone + Ord + PartialEq> PartialEq for Simplex<V> {
  fn eq(&self, other: &Self) -> bool {
    let mut sorted_self = self.vertices.clone();
    sorted_self.sort();
    let mut sorted_other = other.vertices.clone();
    sorted_other.sort();
    sorted_self.iter().zip(sorted_other.iter()).all(|(a, b)| a == b)
  }
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

  pub fn dimension(&self) -> usize {
    self.dimension
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

pub struct SimplicialComplex<V> {
  simplices: Vec<Vec<Simplex<V>>>,
}

impl<V: Clone + PartialEq + Ord> SimplicialComplex<V> {
  pub fn new() -> Self {
    Self { simplices: vec![] }
  }

  pub fn add_simplex(&mut self, simplex: Simplex<V>) {
    while self.simplices.len() <= simplex.dimension() {
      self.simplices.push(Vec::new());
    }
    if self.simplices[simplex.dimension()].contains(&simplex) {
      return;
    }
    self.simplices[simplex.dimension()].push(simplex.clone());

    if simplex.dimension() > 0 {
      for face in simplex.faces() {
        self.add_simplex(face);
      }
    }
  }
}

pub struct Chain<R, V> {
  simplices: Vec<Simplex<V>>,
  coefficients: Vec<R>,
}

impl<R: Add + Mul<Output = R>, V: Clone + PartialEq + Ord> Chain<R, V> {
  pub fn new(simplices: Vec<Simplex<V>>, coefficients: Vec<R>) -> Self {
    Self { simplices, coefficients }
  }
}

impl<R: Add + Mul<Output = R>, V: Clone + PartialEq + Ord> PartialEq for Chain<R, V> {
  fn eq(&self, other: &Self) -> bool {
    self.simplices == other.simplices && self.coefficients == other.coefficients
  }
}

// impl<R: Add + Mul<Output = R>, V: Clone + PartialEq + Ord> Add for Chain<R, V> {
//   type Output = Chain<R, V>;

//   fn add(self, other: Self) -> Self::Output {
//     Self::new(self.simplices + other.simplices, self.coefficients + other.coefficients)
//   }
// }

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

  #[test]
  fn test_simplicial_complex() {
    let mut complex = SimplicialComplex::new();
    complex.add_simplex(Simplex::new(2, vec![0, 1, 2]));
    assert_eq!(complex.simplices[2].len(), 1);
    assert_eq!(complex.simplices[1].len(), 3);
    assert_eq!(complex.simplices[0].len(), 3);
  }
}
