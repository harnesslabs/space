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
  type InputParameter;
  /// The type of the output space (e.g., a simplicial complex).
  type OutputSpace;
  /// The type of the output parameter (e.g., the dimension of the homology group).
  type OutputParameter;

  /// Constructs an output space from the input space using the given parameter.
  ///
  /// # Arguments
  /// * `input`: A reference to the input space.
  /// * `param`: The parameter value that drives the construction.
  ///
  /// # Returns
  /// The constructed output space.
  fn build(
    &self,
    input: &Self::InputSpace,
    input_param: Self::InputParameter,
    output_param: &Self::OutputParameter,
  ) -> Self::OutputSpace;

  /// Builds the output space in serial for multiple parameters.
  ///
  /// # Arguments
  /// * `input`: A reference to the input space.
  /// * `param`: A vector of parameters to build the output space for.
  ///
  /// # Returns
  fn build_serial(
    &self,
    input: &Self::InputSpace,
    input_param: Vec<Self::InputParameter>,
    output_param: &Self::OutputParameter,
  ) -> Vec<Self::OutputSpace> {
    input_param.into_iter().map(|p| self.build(input, p, output_param)).collect()
  }
}

#[cfg(feature = "parallel")] use rayon::prelude::*;

#[cfg(feature = "parallel")]
/// A trait for filtrations that can be built in parallel.
pub trait ParallelFiltration: Filtration
where
  Self: Sync,
  Self::InputSpace: Sync,
  Self::InputParameter: Send,
  Self::OutputSpace: Send,
  Self::OutputParameter: Send + Sync, {
  /// Builds the output space in parallel for multiple parameters.
  ///
  /// # Arguments
  /// * `input`: A reference to the input space.
  /// * `param`: A vector of parameters to build the output space for.
  ///
  /// # Returns
  fn build_parallel(
    &self,
    input: &Self::InputSpace,
    input_param: Vec<Self::InputParameter>,
    output_param: &Self::OutputParameter,
  ) -> Vec<Self::OutputSpace> {
    input_param.into_par_iter().map(|p| self.build(input, p, output_param)).collect()
  }
}
