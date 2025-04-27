use std::ops::{Add, Mul, Neg};

use itertools::Itertools;

// NOTE: Vertices are always stored sorted.
#[derive(Clone, Debug)]
pub struct Simplex<V> {
  vertices: Vec<V>,
  dimension: usize,
}

impl<V: Clone + Ord + PartialEq> PartialEq for Simplex<V> {
  fn eq(&self, other: &Self) -> bool {
    self.vertices == other.vertices
  }
}

impl<V> Simplex<V> {
  // Create a new simplex from K+1 vertices
  pub fn new(dimension: usize, vertices: Vec<V>) -> Self
  where
    V: Ord,
  {
    assert!(vertices.iter().combinations(2).all(|v| v[0] != v[1]));
    assert!(vertices.len() == dimension + 1);
    Self { vertices: vertices.into_iter().sorted().collect(), dimension }
  }

  // Get a reference to the vertices
  pub fn vertices(&self) -> &[V] {
    &self.vertices
  }

  pub fn dimension(&self) -> usize {
    self.dimension
  }

  pub fn faces(&self) -> Vec<Simplex<V>>
  where
    V: Clone + Ord,
  {
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

impl<R, V> Chain<R, V> {
  pub fn new() -> Self {
    Self { simplices: vec![], coefficients: vec![] }
  }

  pub fn from_simplex_and_coeff(simplex: Simplex<V>, coeff: R) -> Self {
    Self { simplices: vec![simplex], coefficients: vec![coeff] }
  }

  pub fn boundary(&self) -> Self
  where
    R: Clone + Neg<Output = R> + Add<Output = R>,
    V: Clone + Ord,
  {
    let mut boundary = Self::new();
    for (coeff, simplex) in self.coefficients.clone().into_iter().zip(self.simplices.iter()) {
      for i in 0..simplex.dimension() {
        let mut vertices = simplex.vertices().to_vec();
        vertices.remove(i);
        // Get the sign of the permutation of the vertices
        let permutation = permutation_sign(&vertices);
        // Make a new simplex with the remaining vertices
        let face = Simplex::new(simplex.dimension() - 1, vertices);
        // Make a new chain with the face and the coefficient where the sign of the permutation is applied
        let chain = Self::from_simplex_and_coeff(
          face,
          if permutation == Permutation::Even { coeff.clone() } else { -coeff.clone() },
        );
        // Add the chain to the boundary
        boundary = boundary + chain;
      }
    }
    boundary
  }
}

impl<R: Clone + Neg + PartialEq, V: Clone + PartialEq + Ord> PartialEq for Chain<R, V> {
  fn eq(&self, other: &Self) -> bool {
    let self_chain = self.coefficients.clone().into_iter().zip(self.simplices.iter());
    let other_chain = other.coefficients.clone().into_iter().zip(other.simplices.iter());

    self_chain.zip(other_chain).all(|((coeff_a, simplex_a), (coeff_b, simplex_b))| {
      coeff_a == coeff_b
        && permutation_sign(simplex_a.vertices()) == permutation_sign(simplex_b.vertices())
        && simplex_a.vertices() == simplex_b.vertices()
    })
  }
}

impl<R: Add + Neg, V: Clone + PartialEq + Ord> Add for Chain<R, V> {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    todo!()
  }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Permutation {
  Odd,
  Even,
}

pub fn permutation_sign<V: Ord>(item: &[V]) -> Permutation {
  let mut count = 0;
  for i in 0..item.len() {
    for j in i + 1..item.len() {
      if item[i] > item[j] {
        count += 1;
      }
    }
  }
  if count % 2 == 0 {
    Permutation::Even
  } else {
    Permutation::Odd
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
