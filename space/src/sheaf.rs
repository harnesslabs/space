use crate::definitions::{Set, TopologicalSpace};

/// A trait representing a cellular sheaf over a topological space.
///
/// A sheaf assigns a vector space (stalk) to each cell and provides
/// restriction maps between stalks when cells are incident.
pub trait Presheaf<T: TopologicalSpace> {
  /// The type of the stalk (vector space) at each point
  type Stalk: Stalk;
  /// The type of sections over open sets
  type Section: Section<T, Stalk = Self::Stalk>;

  /// Restricts a section from a larger open set to a smaller one
  fn restrict(
    &self,
    section: &Self::Section,
    from: &<T as TopologicalSpace>::OpenSet,
    to: &<T as TopologicalSpace>::OpenSet,
  ) -> Self::Section;
}

/// A trait representing a stalk (vector space) in a sheaf
pub trait Stalk {
  /// The type of elements in the stalk
  type Element: Set;
}

pub trait Section<T: TopologicalSpace> {
  /// The type of the stalk this section takes values in
  type Stalk: Stalk;

  /// Evaluates the section at a point in its domain, returning an element of the stalk
  fn evaluate(
    &self,
    point: &<T as TopologicalSpace>::Point,
  ) -> Option<<Self::Stalk as Stalk>::Element>;

  /// Gets the open set over which this section is defined
  fn base(&self) -> <T as TopologicalSpace>::Point;
}
