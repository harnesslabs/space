//! Catch-all enum for all spaces errors.
//! This enum derives std::error::Error using `thiserror`.

use thiserror::Error;

/// Error type for all space constructions
#[derive(Clone, Debug, Error)]
pub enum SpacesError {
  /// Provided object dimension does not match the expectation!
  #[error("Dimension Mismatch")]
  DimensionMismatch,
  /// CW complex is not yet initialized but a function was called assuming it was.
  #[error("Uninitialized CW Complex")]
  CWUninitialized,
  /// Provided `cell_idx` input outside expected range
  #[error("`cell_idx` provided outside of range")]
  InvalidCellIdx,
  /// Failed to fetch a particular `Point` from the CW complex
  #[error("Provided point not found")]
  NoPointFound,
}
