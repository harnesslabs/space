// space/src/filtrations.rs
//! Defines traits for building filtered topological spaces.

pub mod vietoris_rips;

/// A trait for processes that construct an output space from an input space
/// based on a given parameter. This is a core concept in filtrations,
/// where the parameter (e.g., a distance threshold) determines the structure
/// of the output space.
pub trait Filtration {
  /// The type of the input space (e.g., a point cloud).
  type InputSpace;
  /// The type of the parameter used for filtering (e.g., a distance epsilon).
  type Parameter;
  /// The type of the output space (e.g., a simplicial complex).
  type OutputSpace;

  /// Constructs an output space from the input space using the given parameter.
  ///
  /// # Arguments
  /// * `input`: A reference to the input space.
  /// * `param`: The parameter value that drives the construction.
  ///
  /// # Returns
  /// The constructed output space.
  fn build(&self, input: &Self::InputSpace, param: Self::Parameter) -> Self::OutputSpace;
}
