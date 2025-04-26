pub trait Set {
  type Point;
  fn contains(&self, point: Self::Point) -> bool;
  fn difference(&self, other: Self) -> Self;
  fn intersect(&self, other: Self) -> Self;
  fn union(&self, other: Self) -> Self;
}

pub trait TopologicalSpace {
  type Point;
  type OpenSet: Set<Point = Self::Point>;
  fn neighborhood(&self, point: Self::Point) -> Self::OpenSet;
  fn is_open(&self, open_set: Self::OpenSet) -> bool;
}

pub trait MetricSpace: TopologicalSpace {
  type Distance;
  fn distance(
    &self,
    point_a: <Self as TopologicalSpace>::Point,
    point_b: <Self as TopologicalSpace>::Point,
  ) -> Self::Distance;
}

pub trait NormedSpace: MetricSpace {
  type Norm;
  fn norm(&self, point: Self::Point) -> Self::Norm;
}

pub trait InnerProductSpace: NormedSpace {
  type InnerProduct;
  fn inner_product(&self, point_a: Self::Point, point_b: Self::Point) -> Self::InnerProduct;
}
