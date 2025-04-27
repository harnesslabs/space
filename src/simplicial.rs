pub struct Simplex<V, const K: usize>
where [(); K + 1]: {
  // A simplex of dimension K has K+1 vertices
  vertices: [V; K + 1],
}

impl<V: Clone + PartialEq, const K: usize> Simplex<V, K>
where [(); K + 1]:
{
  // Create a new simplex from K+1 vertices
  pub fn new(vertices: [V; K + 1]) -> Self { Self { vertices } }

  // Get a reference to the vertices
  pub fn vertices(&self) -> &[V; K + 1] { &self.vertices }

  pub fn boundary(&self) -> Vec<Simplex<V, { K - 1 }>>
  where [(); (K - 1) + 1]: /* <- THIS is needed for Simplex<V, {K-1}> */ {
    todo!()
  }
}
