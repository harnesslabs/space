use std::{collections::HashSet, hash::Hash, marker::PhantomData};

use crate::definitions::{MetricSpace, Set, TopologicalSpace};

mod sealed {
  pub trait Sealed {}
}

pub trait DirectedType: sealed::Sealed {
  const DIRECTED: bool;
}

pub struct Undirected;
impl sealed::Sealed for Undirected {}
impl DirectedType for Undirected {
  const DIRECTED: bool = false;
}
pub struct Directed;
impl sealed::Sealed for Directed {}
impl DirectedType for Directed {
  const DIRECTED: bool = true;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphPoint<V> {
  Vertex(V),
  EdgePoint(V, V),
}

#[derive(Debug, Clone)]
pub struct Graph<V, D: DirectedType> {
  vertices: HashSet<V>,
  edges: HashSet<(V, V)>,
  d: PhantomData<D>,
}

impl<V: PartialOrd + Eq + Hash, D: DirectedType> Graph<V, D> {
  pub fn new(vertices: HashSet<V>, edges: HashSet<(V, V)>) -> Self {
    let edges = if D::DIRECTED {
      edges
    } else {
      edges.into_iter().map(|(a, b)| if a <= b { (a, b) } else { (b, a) }).collect::<HashSet<_>>()
    };

    assert!(
      edges.iter().all(|(a, b)| vertices.contains(a) && vertices.contains(b)),
      "All edges must be between vertices",
    );
    Self { vertices, edges, d: PhantomData }
  }
}

impl<V: PartialOrd + Eq + Hash + Clone, D: DirectedType> Set for Graph<V, D> {
  type Point = GraphPoint<V>;

  fn contains(&self, point: &Self::Point) -> bool {
    match point {
      GraphPoint::Vertex(v) => self.vertices.contains(&v),
      GraphPoint::EdgePoint(u, v) => {
        self.edges.contains(&(u.clone(), v.clone())) | self.edges.contains(&(v.clone(), u.clone()))
      },
    }
  }

  fn difference(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.difference(&other.vertices).cloned().collect();

    let edges: HashSet<(V, V)> = self
      .edges
      .iter()
      .filter(|(u, v)| {
        // Keep edge if both vertices are in our result
        self.vertices.contains(u)
          && self.vertices.contains(v)
          && !other.edges.contains(&(u.clone(), v.clone()))
      })
      .cloned()
      .collect();

    Self::new(vertices, edges)
  }

  fn intersect(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.intersection(&other.vertices).cloned().collect();

    let edges: HashSet<(V, V)> = self.edges.intersection(&other.edges).cloned().collect();

    Self::new(vertices, edges)
  }

  fn union(&self, other: &Self) -> Self {
    let vertices: HashSet<V> = self.vertices.union(&other.vertices).cloned().collect();

    let edges: HashSet<(V, V)> = self.edges.union(&other.edges).cloned().collect();

    Self::new(vertices, edges)
  }
}

// impl<V: PartialOrd + Eq + Hash + Clone, D: DirectedType> TopologicalSpace for Graph<V, D> {
//   type Point = GraphPoint<V>;

//   type OpenSet = HashSet<GraphPoint<V>>;

//   fn neighborhood(&self, point: Self::Point) -> Self::OpenSet {
//     self
//       .edges
//       .iter()
//       .filter_map(|(a, b)| {
//         if a == point {
//           Some(*b)
//         } else if b == point {
//           Some(*a)
//         } else {
//           None
//         }
//       })
//       .collect()
//   }

//   fn is_open(&self, _set: Self::OpenSet) -> bool {
//     true
//   }
// }

// impl MetricSpace for UndirectedGraph {
//   type Distance = Option<usize>;

//   fn distance(
//     &self,
//     point_a: <Self as TopologicalSpace>::Point,
//     point_b: <Self as TopologicalSpace>::Point,
//   ) -> Self::Distance {
//     let mut visited = HashSet::new();
//     let mut queue = vec![(point_a, 0)];
//     while let Some((point, distance)) = queue.pop() {
//       if point == point_b {
//         return Some(distance);
//       }
//       visited.insert(point);
//       for neighbor in self.neighborhood(point) {
//         if !visited.contains(&neighbor) {
//           queue.push((neighbor, distance + 1));
//         }
//       }
//     }
//     None
//   }
// }

#[cfg(test)]
mod tests {

  use super::*;

  fn create_graph() -> Graph<usize, Undirected> {
    let mut vertices = HashSet::new();
    vertices.insert(1);
    vertices.insert(2);
    vertices.insert(3);
    vertices.insert(4);
    vertices.insert(5);

    let mut edges = HashSet::new();
    edges.insert((1, 2));
    edges.insert((2, 3));
    edges.insert((3, 4));

    Graph::new(vertices, edges)
  }

  #[test]
  fn graph_builds() {
    let graph = create_graph();
    assert_eq!(graph.vertices.len(), 5);
    assert_eq!(graph.edges.len(), 3);
  }

  //   #[test]
  //   fn neighborhood() {
  //     let graph = create_graph();
  //     assert_eq!(graph.neighborhood(1), vec![2].into_iter().collect::<HashSet<_>>());
  //     assert_eq!(graph.neighborhood(2), vec![1, 3].into_iter().collect::<HashSet<_>>());
  //     assert_eq!(graph.neighborhood(3), vec![2, 4].into_iter().collect::<HashSet<_>>());
  //     assert_eq!(graph.neighborhood(4), vec![3].into_iter().collect::<HashSet<_>>());
  //   }

  //   #[test]
  //   fn distance() {
  //     let graph = create_graph();
  //     assert_eq!(graph.distance(1, 1), Some(0));
  //     assert_eq!(graph.distance(1, 2), Some(1));
  //     assert_eq!(graph.distance(1, 3), Some(2));
  //     assert_eq!(graph.distance(1, 4), Some(3));
  //     assert_eq!(graph.distance(1, 5), None);
  //   }
}
