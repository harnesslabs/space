#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod cloud;
pub mod complexes;
pub mod definitions;
pub mod filtration;
pub mod graph;
pub mod homology;
pub mod lattice;
pub mod set;
pub mod sheaf;

use harness_algebra::prelude::*;

pub mod prelude {
  //! The prelude for the `space` crate.
  //!
  //! This module re-exports the most commonly used types and traits from the `space` crate.
  //! It provides a convenient way to import these types and traits into your code without
  //! having to specify the crate name each time.
  pub use crate::{
    definitions::{MetricSpace, NormedSpace, TopologicalSpace},
    set::{Collection, Set},
  };
}
