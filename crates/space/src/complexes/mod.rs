//! # Complexes Module
//!
//! This module provides data structures and algorithms for working with
//! various types of topological complexes, primarily focusing on cell complexes
//! and simplicial complexes. These are fundamental tools in algebraic topology
//! for representing and analyzing the structure of topological spaces.
//!
//! ## Submodules
//! - [`cell`]: Contains definitions for `Cell` and `CellComplex`, allowing for the construction and
//!   manipulation of regular cell complexes.
//! - [`simplicial`]: Contains definitions for `Simplex` and `SimplicialComplex`, providing tools
//!   for working with simplicial topology, including homology computations.

use super::*;

pub mod cell;
pub mod simplicial;
