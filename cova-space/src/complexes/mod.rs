//! # Complexes Module
//!
//! This module provides data structures and algorithms for working with various types of
//! topological complexes, which are fundamental tools in algebraic topology for representing
//! and analyzing the structure of topological spaces.
//!
//! ## Mathematical Background
//!
//! A **cell complex** (or CW complex) is a topological space constructed by gluing together
//! cells of various dimensions. The key insight is that complex topological spaces can be
//! built up systematically from simple pieces:
//!
//! - **0-cells**: Points (vertices)
//! - **1-cells**: Edges connecting vertices
//! - **2-cells**: Faces (triangles, squares) with edges as boundaries
//! - **k-cells**: Higher-dimensional analogs
//!
//! This module focuses on two important special cases:
//! - **Simplicial complexes**: Built from simplices (points, edges, triangles, tetrahedra, ...)
//! - **Cubical complexes**: Built from cubes (points, edges, squares, cubes, ...)
//!
//! ## Core Abstractions
//!
//! ### Generic Complex Structure
//!
//! The [`Complex<T>`] type provides a generic container that can work with any element type
//! implementing [`ComplexElement`]. This allows the same algorithms (homology computation,
//! boundary operators, etc.) to work on both simplicial and cubical complexes.
//!
//! ### Face Relations and Closure Property
//!
//! All complexes satisfy the **closure property**: if a k-cell is in the complex, then all
//! of its faces (boundary cells) must also be in the complex. This is automatically
//! enforced when adding elements via [`Complex::join_element`].
//!
//! ### Orientation and Boundary Operators
//!
//! Each element type implements its own boundary operator via
//! [`ComplexElement::boundary_with_orientations`], which returns faces with their correct
//! orientation coefficients. This enables:
//! - Computation of boundary matrices for homology
//! - Chain complex structure (∂² = 0)
//! - Proper handling of orientation in both simplicial and cubical settings
//!
//! ## Usage Patterns
//!
//! ### Basic Complex Construction
//!
//! ```rust
//! use cova_space::complexes::{Complex, Simplex};
//!
//! let mut complex = Complex::new();
//!
//! // Add a triangle - automatically includes all faces
//! let triangle = Simplex::new(2, vec![0, 1, 2]);
//! let added_triangle = complex.join_element(triangle);
//!
//! // Complex now contains: 1 triangle, 3 edges, 3 vertices
//! assert_eq!(complex.elements_of_dimension(2).len(), 1);
//! assert_eq!(complex.elements_of_dimension(1).len(), 3);
//! assert_eq!(complex.elements_of_dimension(0).len(), 3);
//! ```
//!
//! ### Homology Computation
//!
//! ```rust
//! use cova_algebra::algebras::boolean::Boolean;
//! use cova_space::{
//!   complexes::{Simplex, SimplicialComplex},
//!   prelude::*,
//! };
//!
//! let mut complex = SimplicialComplex::new();
//! let triangle = Simplex::new(2, vec![0, 1, 2]);
//!
//! // Compute homology over Z/2Z
//! let h0 = complex.homology::<Boolean>(0); // Connected components
//! let h1 = complex.homology::<Boolean>(1); // 1D holes
//!
//! println!("β₀ = {}, β₁ = {}", h0.betti_number, h1.betti_number);
//! ```
//!
//! ### Working with Different Element Types
//!
//! ```rust
//! use cova_space::complexes::{Cube, CubicalComplex, Simplex, SimplicialComplex};
//!
//! // Simplicial complex with triangles
//! let mut simplicial = SimplicialComplex::new();
//! let triangle = Simplex::new(2, vec![0, 1, 2]);
//! simplicial.join_element(triangle);
//!
//! // Cubical complex with squares
//! let mut cubical = CubicalComplex::new();
//! let square = Cube::square([0, 1, 2, 3]);
//! cubical.join_element(square);
//!
//! // Both support the same operations
//! assert_eq!(simplicial.max_dimension(), 2);
//! assert_eq!(cubical.max_dimension(), 2);
//! ```
//!
//! ## Implementation Details
//!
//! ### Efficient Storage and Lookup
//!
//! - Elements are stored in a `HashMap<usize, T>` keyed by unique IDs
//! - Face relationships are tracked in a `Lattice<usize>` using IDs for efficiency
//! - Duplicate elements (same mathematical content) are automatically deduplicated
//! - ID assignment is automatic but can be controlled when needed
//!
//! ### Poset Structure
//!
//! Complexes implement [`Poset`] where the partial order represents the face relation:
//! `a ≤ b` means "a is a face of b". This enables:
//! - Efficient computation of upsets/downsets (star/closure operations)
//! - Join/meet operations for common faces/cofaces
//! - Integration with other algebraic structures
//!
//! ### Topology Interface
//!
//! Complexes implement [`Topology`] providing:
//! - Neighborhood operations (finding cofaces)
//! - Boundary operators returning [`Chain`] objects
//! - Integration with homology and sheaf computations
//!
//! ## Submodules
//!
//! - [`simplicial`]: Definitions for [`Simplex`] and simplicial complex operations
//! - [`cubical`]: Definitions for [`Cube`] and cubical complex operations
//!
//! ## Examples
//!
//! See the extensive test suite at the bottom of this module for examples of:
//! - Constructing various complex types
//! - Computing homology of standard spaces
//! - Working with the poset and topology interfaces
//! - Verifying fundamental properties like ∂² = 0

use std::collections::HashMap;

use cova_algebra::{
  rings::Field,
  tensors::dynamic::{compute_quotient_basis, Matrix, Vector},
};

use super::*;
use crate::{
  definitions::Topology,
  homology::{Chain, Homology},
  lattice::Lattice,
  set::{Collection, Poset},
};

pub mod cubical;
pub mod simplicial;

pub use cubical::Cube;
pub use simplicial::Simplex;

/// A type alias for a simplicial complex.
pub type SimplicialComplex = Complex<Simplex>;

/// A type alias for a cubical complex.
pub type CubicalComplex = Complex<Cube>;

/// Trait for elements that can be part of a topological complex.
///
/// This trait captures the essential behavior needed for elements (simplices, cubes, cells, etc.)
/// to work with the generic [`Complex<T>`] structure. It abstracts over the common operations
/// that any type of cell complex element must support.
///
/// # Mathematical Foundation
///
/// In algebraic topology, a complex is built from cells of various dimensions with specific
/// face relations. This trait encapsulates the core properties that any cell type must have:
///
/// 1. **Dimension**: Each cell has an intrinsic dimension (0 for vertices, 1 for edges, etc.)
/// 2. **Boundary Structure**: Each cell has well-defined boundary cells (faces)
/// 3. **Orientation**: Boundary relationships include orientation information for chain complexes
/// 4. **Identity**: Cells can be uniquely identified when added to complexes
/// 5. **Content Equality**: Mathematical content can be compared regardless of ID assignment
///
/// # Design Philosophy
///
/// This trait is designed to be:
/// - **Generic**: Works with simplices, cubes, and other cell types
/// - **Efficient**: Supports ID-based operations for large complexes
/// - **Mathematically Correct**: Preserves orientation and boundary relationships
/// - **Flexible**: Allows different boundary operator conventions per element type
///
/// # Implementation Requirements
///
/// Types implementing this trait must:
/// - Be cloneable, hashable, and orderable for use in collections
/// - Compute their own faces and boundary operators correctly
/// - Handle ID assignment and content comparison properly
/// - Maintain mathematical consistency in their face relationships
///
/// # Examples
///
/// ```rust
/// use cova_space::complexes::{ComplexElement, Simplex};
///
/// // Create a triangle (2-simplex)
/// let triangle = Simplex::new(2, vec![0, 1, 2]);
///
/// // Basic properties
/// assert_eq!(triangle.dimension(), 2);
/// assert_eq!(triangle.id(), None); // No ID until added to complex
///
/// // Compute faces (should be 3 edges)
/// let faces = triangle.faces();
/// assert_eq!(faces.len(), 3);
/// assert!(faces.iter().all(|face| face.dimension() == 1));
///
/// // Compute boundary with orientations
/// let boundary = triangle.boundary_with_orientations();
/// assert_eq!(boundary.len(), 3);
/// // Each face has orientation ±1
/// assert!(boundary.iter().all(|(_, orient)| orient.abs() == 1));
/// ```
pub trait ComplexElement: Clone + std::hash::Hash + Eq + PartialOrd + Ord {
  /// Returns the intrinsic dimension of this element.
  ///
  /// The dimension determines the element's place in the chain complex:
  /// - 0-dimensional: vertices/points
  /// - 1-dimensional: edges/curves
  /// - 2-dimensional: faces/surfaces
  /// - k-dimensional: k-cells
  ///
  /// # Mathematical Note
  ///
  /// For a k-dimensional element, its faces are (k-1)-dimensional, and it can be
  /// a face of (k+1)-dimensional elements. This creates the graded structure
  /// essential for homological computations.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{ComplexElement, Simplex, Cube};
  /// let vertex = Simplex::new(0, vec![42]);
  /// assert_eq!(vertex.dimension(), 0);
  ///
  /// let edge = Cube::edge(0, 1);
  /// assert_eq!(edge.dimension(), 1);
  /// ```
  fn dimension(&self) -> usize;

  /// Returns all faces (boundary elements) of this element.
  ///
  /// For a k-dimensional element, this returns all (k-1)-dimensional faces that
  /// form its boundary. This is the **combinatorial boundary** - it captures the
  /// face structure without orientation information.
  ///
  /// # Mathematical Background
  ///
  /// In topology, the boundary ∂σ of a cell σ consists of all the cells in its
  /// boundary. For example:
  /// - Triangle faces: the three edges forming its boundary
  /// - Tetrahedron faces: the four triangular faces
  /// - Square faces: the four edges forming its boundary
  ///
  /// # Implementation Notes
  ///
  /// - Returned faces should have no ID assigned (will be assigned when added to complex)
  /// - The order may matter for orientation in [`ComplexElement::boundary_with_orientations`]
  /// - All faces must have dimension = `self.dimension() - 1`
  /// - 0-dimensional elements return empty vector (no (-1)-dimensional faces)
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{ComplexElement, Simplex};
  /// // Triangle has 3 edge faces
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// let faces = triangle.faces();
  /// assert_eq!(faces.len(), 3);
  /// assert!(faces.iter().all(|f| f.dimension() == 1));
  ///
  /// // Vertex has no faces
  /// let vertex = Simplex::new(0, vec![0]);
  /// assert_eq!(vertex.faces().len(), 0);
  /// ```
  fn faces(&self) -> Vec<Self>;

  /// Returns the faces with their correct orientation coefficients for boundary computation.
  ///
  /// This is the **geometric boundary operator** that includes orientation information
  /// necessary for chain complex computations. Each face comes with an integer coefficient
  /// (typically ±1) indicating its orientation in the boundary.
  ///
  /// # Mathematical Foundation
  ///
  /// In algebraic topology, the boundary operator ∂ₖ: Cₖ → Cₖ₋₁ is defined as:
  ///
  /// ```text
  /// ∂ₖ(σ) = Σᵢ (-1)ⁱ τᵢ
  /// ```
  ///
  /// where the τᵢ are the faces of σ with appropriate orientation signs. The key
  /// property is that ∂² = 0 (boundary of boundary is zero), which requires
  /// careful orientation handling.
  ///
  /// # Orientation Conventions
  ///
  /// Different element types may use different orientation conventions:
  /// - **Simplicial**: Alternating signs based on vertex position
  /// - **Cubical**: Signs based on coordinate directions
  /// - **General CW**: Depends on attaching maps
  ///
  /// # Return Format
  ///
  /// Returns `Vec<(face, orientation)>` where:
  /// - `face`: A (k-1)-dimensional face element (without ID)
  /// - `orientation`: Integer coefficient (usually ±1, could be 0)
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{ComplexElement, Simplex};
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// let boundary = triangle.boundary_with_orientations();
  ///
  /// // Triangle boundary: [v₁,v₂] - [v₀,v₂] + [v₀,v₁]
  /// assert_eq!(boundary.len(), 3);
  /// assert_eq!(boundary[0].1, 1); // +[v₁,v₂]
  /// assert_eq!(boundary[1].1, -1); // -[v₀,v₂]
  /// assert_eq!(boundary[2].1, 1); // +[v₀,v₁]
  /// ```
  fn boundary_with_orientations(&self) -> Vec<(Self, i32)>;

  /// Returns the ID if this element has been assigned to a complex, `None` otherwise.
  ///
  /// IDs are automatically assigned when elements are added to a [`Complex`] via
  /// [`Complex::join_element`]. They serve as unique identifiers for efficient
  /// storage and lookup operations.
  ///
  /// # ID Assignment Lifecycle
  ///
  /// 1. **Created**: Element starts with `id() = None`
  /// 2. **Added**: Complex assigns unique ID via [`ComplexElement::with_id`]
  /// 3. **Stored**: Element with ID is stored in complex's HashMap
  /// 4. **Referenced**: ID used for lattice operations and lookups
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// let simplex = Simplex::new(1, vec![0, 1]);
  /// assert_eq!(simplex.id(), None);
  ///
  /// let mut complex = Complex::new();
  /// let added = complex.join_element(simplex);
  /// assert!(added.id().is_some());
  /// ```
  fn id(&self) -> Option<usize>;

  /// Checks if this element has the same mathematical content as another.
  ///
  /// This comparison ignores ID assignment and focuses purely on the mathematical
  /// structure of the elements. It's used for deduplication when adding elements
  /// to complexes.
  ///
  /// # Mathematical Equality
  ///
  /// Two elements are considered to have the same content if they represent
  /// the same mathematical object, regardless of:
  /// - ID assignment (internal bookkeeping)
  /// - Order of discovery (when added to complex)
  /// - Memory location or other implementation details
  ///
  /// # Usage in Complexes
  ///
  /// When [`Complex::join_element`] is called, it first checks if an element
  /// with the same content already exists using this method. If found, it
  /// returns the existing element rather than creating a duplicate.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{ComplexElement, Simplex};
  /// let simplex1 = Simplex::new(1, vec![0, 1]);
  /// let simplex2 = Simplex::new(1, vec![0, 1]);
  /// let simplex3 = simplex1.clone().with_id(42);
  ///
  /// assert!(simplex1.same_content(&simplex2)); // Same mathematical content
  /// assert!(simplex1.same_content(&simplex3)); // ID differences ignored
  ///
  /// let different = Simplex::new(1, vec![0, 2]);
  /// assert!(!simplex1.same_content(&different)); // Different content
  /// ```
  fn same_content(&self, other: &Self) -> bool;

  /// Creates a new element with the same content but a specific ID.
  ///
  /// This is used internally by [`Complex`] to assign IDs when adding elements.
  /// The returned element should be identical in all mathematical properties
  /// but have the specified ID assigned.
  ///
  /// # Implementation Requirements
  ///
  /// The returned element must satisfy:
  /// - `result.same_content(self) == true`
  /// - `result.id() == Some(new_id)`
  /// - All other properties unchanged (dimension, faces, etc.)
  ///
  /// # Usage
  ///
  /// This method is primarily used internally by the complex management system.
  /// Users typically don't need to call it directly.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{ComplexElement, Simplex};
  /// let original = Simplex::new(1, vec![0, 1]);
  /// assert_eq!(original.id(), None);
  ///
  /// let with_id = original.with_id(42);
  /// assert_eq!(with_id.id(), Some(42));
  /// assert!(original.same_content(&with_id));
  /// assert_eq!(original.dimension(), with_id.dimension());
  /// ```
  fn with_id(&self, new_id: usize) -> Self;
}

/// A generic topological complex that can work with any type implementing [`ComplexElement`].
///
/// This structure provides a unified framework for working with various types of cell complexes
/// (simplicial, cubical, etc.) while maintaining the essential topological and algebraic
/// properties needed for homological computations.
///
/// # Mathematical Foundation
///
/// A **cell complex** K is a topological space built by attaching cells of various dimensions
/// according to specific rules:
///
/// 1. **Closure Property**: If σ ∈ K, then all faces of σ are also in K
/// 2. **Face Relations**: Form a partial order where τ ≤ σ means "τ is a face of σ"
/// 3. **Dimension Stratification**: K = K⁰ ∪ K¹ ∪ K² ∪ ... where Kⁱ contains all i-cells
/// 4. **Chain Complex Structure**: Boundary operators ∂ᵢ: Cᵢ → Cᵢ₋₁ with ∂² = 0
///
/// # Implementation Architecture
///
/// This implementation uses a **dual-structure approach**:
///
/// ```text
/// ┌─────────────────┐    ┌──────────────────┐
/// │ attachment_     │    │ elements:        │
/// │ lattice:        │◄──►│ HashMap<usize,T> │
/// │ Lattice<usize>  │    │                  │  
/// └─────────────────┘    └──────────────────┘
///        ▲                        ▲
///        │ IDs only               │ Full elements
///        ▼                        ▼
///   Fast operations           Rich operations
/// ```
///
/// - **Elements HashMap**: Stores actual elements indexed by unique IDs
/// - **Attachment Lattice**: Tracks face relationships using IDs for efficiency
/// - **ID Management**: Automatic assignment with deduplication support
///
/// # Key Properties
///
/// ## Closure Property Enforcement
///
/// The [`Complex::join_element`] method ensures closure: adding any element automatically
/// includes all its faces. This maintains the fundamental property that distinguishes
/// complexes from arbitrary cell collections.
///
/// ## Efficient Face Queries
///
/// Face relationships are stored in a [`Lattice<usize>`] structure, enabling:
/// - O(1) face/coface lookups after preprocessing
/// - Efficient upset/downset computations
/// - Support for meet/join operations on cells
///
/// ## Deduplication
///
/// Elements with identical mathematical content are automatically deduplicated
/// using [`ComplexElement::same_content`], preventing redundant storage and
/// maintaining well-defined structure.
///
/// # Usage Patterns
///
/// ## Basic Construction
///
/// ```rust
/// use cova_space::complexes::{Complex, Simplex};
///
/// let mut complex = Complex::new();
/// let triangle = Simplex::new(2, vec![0, 1, 2]);
/// let added = complex.join_element(triangle);
///
/// // Automatically includes all faces: 1 triangle + 3 edges + 3 vertices
/// assert_eq!(complex.elements_of_dimension(2).len(), 1);
/// assert_eq!(complex.elements_of_dimension(1).len(), 3);
/// assert_eq!(complex.elements_of_dimension(0).len(), 3);
/// ```
///
/// ## Homology Computation
///
/// ```rust
/// use cova_algebra::algebras::boolean::Boolean;
/// use cova_space::{
///   complexes::{Simplex, SimplicialComplex},
///   prelude::*,
/// };
///
/// let mut complex = SimplicialComplex::new();
///
/// // Create a circle (1-dimensional hole)
/// let edge1 = Simplex::new(1, vec![0, 1]);
/// let edge2 = Simplex::new(1, vec![1, 2]);
/// let edge3 = Simplex::new(1, vec![2, 0]);
/// complex.join_element(edge1);
/// complex.join_element(edge2);
/// complex.join_element(edge3);
///
/// let h1 = complex.homology::<Boolean>(1);
/// assert_eq!(h1.betti_number, 1); // One 1D hole
/// ```
///
/// ## Working with Face Relations
///
/// ```rust
/// use cova_space::{
///   complexes::{Simplex, SimplicialComplex},
///   prelude::*,
/// };
///
/// let mut complex = SimplicialComplex::new();
/// let triangle = Simplex::new(2, vec![0, 1, 2]);
/// let added = complex.join_element(triangle);
///
/// // Query face relationships
/// let faces = complex.faces(&added); // Direct faces only
/// let cofaces = complex.cofaces(&added); // Direct cofaces only
/// let downset = complex.downset(added); // All faces (transitive)
/// ```
///
/// # Performance Characteristics
///
/// - **Element Access**: O(1) by ID, O(n) by content search
/// - **Face Queries**: O(1) for direct faces, O(k) for k-dimensional queries
/// - **Adding Elements**: O(f) where f is the number of faces to add
/// - **Homology**: O(n³) where n is the number of elements in relevant dimensions
///
/// # Type Parameters
///
/// * `T`: The element type, must implement [`ComplexElement`]
///
/// # Examples
///
/// See the extensive test suite for examples including:
/// - Construction of standard complexes (simplicial, cubical)
/// - Homology computations for various topological spaces
/// - Integration with poset and topology interfaces
#[derive(Debug, Clone)]
pub struct Complex<T: ComplexElement> {
  /// The attachment relationships between elements, represented as a lattice of element IDs.
  ///
  /// This lattice encodes the face relation: if element `a` is a face of element `b`,
  /// then `attachment_lattice.leq(a.id(), b.id())` returns `Some(true)`.
  ///
  /// Using IDs rather than full elements provides:
  /// - **Memory efficiency**: IDs are much smaller than full elements
  /// - **Performance**: Integer comparisons are faster than element comparisons
  /// - **Flexibility**: Lattice operations independent of element type
  pub attachment_lattice: Lattice<usize>,

  /// A map storing all elements in the complex, keyed by their assigned ID.
  ///
  /// This provides:
  /// - **Fast lookup**: O(1) access to elements by ID
  /// - **Rich operations**: Access to full element data and methods
  /// - **Content queries**: Iteration over elements by dimension, type, etc.
  pub elements: HashMap<usize, T>,

  /// The counter for the next available unique element identifier.
  ///
  /// This ensures that:
  /// - Each element gets a unique ID when added
  /// - IDs are assigned sequentially for predictable behavior
  /// - The complex can manage arbitrary numbers of elements
  pub next_id: usize,
}

impl<T: ComplexElement> Complex<T> {
  /// Creates a new, empty complex.
  ///
  /// The empty complex contains no elements and has trivial homology:
  /// - H₀ = 0 (no connected components)
  /// - Hₖ = 0 for all k > 0 (no higher-dimensional features)
  ///
  /// # Examples
  ///
  /// ```rust
  /// use cova_space::{
  ///   complexes::{Complex, Simplex},
  ///   prelude::*,
  /// };
  ///
  /// let complex: Complex<Simplex> = Complex::new();
  /// assert!(complex.is_empty());
  /// assert_eq!(complex.max_dimension(), 0);
  /// ```
  pub fn new() -> Self {
    Self {
      attachment_lattice: Lattice::new(),
      elements:           HashMap::new(),
      next_id:            0,
    }
  }

  /// Adds an element to the complex along with all its faces.
  ///
  /// This is the **fundamental operation** for building complexes. It ensures the closure
  /// property by recursively adding all faces of the given element. If an element with
  /// equivalent mathematical content already exists, returns the existing element without
  /// modification.
  ///
  /// # Mathematical Foundation
  ///
  /// In topology, a complex must satisfy the **closure property**:
  ///
  /// > If σ ∈ K, then ∂σ ⊆ K
  ///
  /// This method enforces this property by:
  /// 1. Computing all faces of the input element via [`ComplexElement::faces`]
  /// 2. Recursively adding each face (which adds their faces, etc.)
  /// 3. Establishing face relationships in the attachment lattice
  /// 4. Deduplicating based on mathematical content
  ///
  /// # Algorithm
  ///
  /// ```text
  /// join_element(σ):
  ///   1. Check if equivalent element already exists → return existing
  ///   2. Assign ID to σ (reuse existing ID if valid, else assign next_id)
  ///   3. For each face τ in faces(σ):
  ///        added_τ ← join_element(τ)  // Recursive call
  ///        Add relation: added_τ ≤ σ to lattice
  ///   4. Store σ in elements map
  ///   5. Return σ with assigned ID
  /// ```
  ///
  /// # ID Assignment Strategy
  ///
  /// - **No ID**: Assigns `next_id` and increments counter
  /// - **Has unused ID**: Preserves existing ID if not taken
  /// - **Has conflicting ID**: Assigns new ID to avoid conflicts
  ///
  /// # Deduplication Logic
  ///
  /// Uses [`ComplexElement::same_content`] to check for existing equivalent elements.
  /// This ensures that mathematically identical elements (regardless of ID) are not
  /// duplicated in the complex.
  ///
  /// # Face Relationship Management
  ///
  /// Establishes `face_id ≤ element_id` relationships in the attachment lattice for
  /// all direct faces. Transitive relationships are computed automatically by the
  /// lattice structure.
  ///
  /// # Return Value
  ///
  /// Returns the element as it exists in the complex (with assigned ID). This may be:
  /// - The input element with a newly assigned ID
  /// - An existing equivalent element if content matches
  ///
  /// # Performance Notes
  ///
  /// - **Time**: O(f) where f is the total number of faces to add (including recursive)
  /// - **Space**: O(n) additional storage where n is the number of new elements
  /// - **Deduplication**: O(k) check where k is the number of existing elements
  ///
  /// # Examples
  ///
  /// ## Basic Usage
  ///
  /// ```rust
  /// use cova_space::complexes::{Complex, Simplex};
  ///
  /// let mut complex = Complex::new();
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  ///
  /// let added = complex.join_element(triangle);
  /// assert!(added.id().is_some());
  ///
  /// // Complex now contains triangle + 3 edges + 3 vertices
  /// assert_eq!(complex.elements_of_dimension(2).len(), 1);
  /// assert_eq!(complex.elements_of_dimension(1).len(), 3);
  /// assert_eq!(complex.elements_of_dimension(0).len(), 3);
  /// ```
  ///
  /// ## Deduplication
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// let edge1 = Simplex::new(1, vec![0, 1]);
  /// let edge2 = Simplex::new(1, vec![0, 1]); // Same content
  ///
  /// let added1 = complex.join_element(edge1);
  /// let added2 = complex.join_element(edge2);
  ///
  /// // Returns same element (by ID)
  /// assert_eq!(added1.id(), added2.id());
  /// assert_eq!(complex.elements_of_dimension(1).len(), 1);
  /// ```
  ///
  /// ## Building Complex Structures
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// let mut complex = Complex::new();
  ///
  /// // Add multiple triangles sharing edges
  /// let triangle1 = Simplex::new(2, vec![0, 1, 2]);
  /// let triangle2 = Simplex::new(2, vec![1, 2, 3]);
  ///
  /// complex.join_element(triangle1);
  /// complex.join_element(triangle2);
  ///
  /// // Shared edge [1,2] is not duplicated
  /// assert_eq!(complex.elements_of_dimension(2).len(), 2); // 2 triangles
  /// assert_eq!(complex.elements_of_dimension(1).len(), 5); // 5 unique edges
  /// assert_eq!(complex.elements_of_dimension(0).len(), 4); // 4 vertices
  /// ```
  pub fn join_element(&mut self, element: T) -> T {
    // Check if we already have this element (by mathematical content)
    if let Some(existing) = self.find_equivalent_element(&element) {
      return existing;
    }

    // Assign a new ID to this element
    let element_with_id =
      if element.id().is_some() && !self.elements.contains_key(&element.id().unwrap()) {
        // Element already has an ID and it's not taken
        if element.id().unwrap() >= self.next_id {
          self.next_id = element.id().unwrap() + 1;
        }
        element
      } else {
        // Assign a new ID
        let new_id = self.next_id;
        self.next_id += 1;
        element.with_id(new_id)
      };

    let mut face_ids = Vec::new();
    for face in element_with_id.faces() {
      let added_face = self.join_element(face);
      face_ids.push(added_face.id().unwrap()); // Safe because we just added it
    }

    let element_id = element_with_id.id().unwrap();
    self.attachment_lattice.add_element(element_id);

    for face_id in face_ids {
      self.attachment_lattice.add_relation(face_id, element_id);
    }

    self.elements.insert(element_id, element_with_id.clone());
    element_with_id
  }

  /// Finds an element in the complex with equivalent mathematical content.
  ///
  /// This is used internally by [`join_element`] for deduplication. Returns the first
  /// element found with matching content, or `None` if no match exists.
  ///
  /// # Performance
  ///
  /// This performs a linear search through all elements, so it's O(n) where n is the
  /// total number of elements in the complex. For large complexes, this can be a
  /// bottleneck in construction.
  ///
  /// # Future Optimizations
  ///
  /// Potential improvements could include:
  /// - Content-based hashing for faster lookup
  /// - Spatial indexing for geometric elements
  /// - Dimension-stratified search to reduce search space
  fn find_equivalent_element(&self, element: &T) -> Option<T> {
    self.elements.values().find(|existing| element.same_content(existing)).cloned()
  }

  /// Retrieves an element by its unique ID.
  ///
  /// Returns `None` if no element with the given ID exists in the complex.
  /// This is the primary method for accessing elements when you have their ID.
  ///
  /// # Performance
  ///
  /// O(1) HashMap lookup.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// let simplex = Simplex::new(1, vec![0, 1]);
  /// let added = complex.join_element(simplex);
  /// let id = added.id().unwrap();
  ///
  /// let retrieved = complex.get_element(id).unwrap();
  /// assert!(added.same_content(retrieved));
  /// ```
  pub fn get_element(&self, id: usize) -> Option<&T> { self.elements.get(&id) }

  /// Returns all elements of a specific dimension.
  ///
  /// This is useful for:
  /// - Constructing basis sets for homology computations
  /// - Analyzing the dimensional structure of the complex
  /// - Iterating over elements by type (vertices, edges, faces, etc.)
  ///
  /// # Performance
  ///
  /// O(n) where n is the total number of elements, as it must check the dimension
  /// of every element.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// # let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// # complex.join_element(triangle);
  ///
  /// let vertices = complex.elements_of_dimension(0); // All 0-cells
  /// let edges = complex.elements_of_dimension(1); // All 1-cells
  /// let faces = complex.elements_of_dimension(2); // All 2-cells
  ///
  /// assert_eq!(vertices.len(), 3);
  /// assert_eq!(edges.len(), 3);
  /// assert_eq!(faces.len(), 1);
  /// ```
  pub fn elements_of_dimension(&self, dimension: usize) -> Vec<T> {
    self.elements.values().filter(|element| element.dimension() == dimension).cloned().collect()
  }

  /// Returns the maximum dimension of any element in the complex.
  ///
  /// For an empty complex, returns 0. This is useful for determining the
  /// "dimension" of the complex as a topological space.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// assert_eq!(complex.max_dimension(), 0); // Empty complex
  ///
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// complex.join_element(triangle);
  /// assert_eq!(complex.max_dimension(), 2); // 2D complex
  /// ```
  pub fn max_dimension(&self) -> usize {
    self.elements.values().map(ComplexElement::dimension).max().unwrap_or(0)
  }

  /// Returns the direct faces of an element within this complex.
  ///
  /// This differs from [`ComplexElement::faces`] in that it returns elements that
  /// actually exist in the complex (with assigned IDs) rather than abstract face
  /// descriptions.
  ///
  /// # Relationship to Lattice Operations
  ///
  /// This is equivalent to finding the immediate predecessors of the element in
  /// the face lattice, but returns the full element objects rather than just IDs.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// let added_triangle = complex.join_element(triangle);
  ///
  /// let faces = complex.faces(&added_triangle);
  /// assert_eq!(faces.len(), 3); // Three edges
  /// assert!(faces.iter().all(|face| face.dimension() == 1));
  /// assert!(faces.iter().all(|face| face.id().is_some()));
  /// ```
  pub fn faces(&self, element: &T) -> Vec<T> {
    element.id().map_or_else(Vec::new, |id| {
      self
        .attachment_lattice
        .predecessors(id)
        .into_iter()
        .filter_map(|face_id| self.get_element(face_id).cloned())
        .collect()
    })
  }

  /// Returns the direct cofaces of an element within this complex.
  ///
  /// Cofaces are elements that have the given element as a face. This is the
  /// "upward" direction in the face lattice.
  ///
  /// # Examples
  ///
  /// ```rust
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// # let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// # let added_triangle = complex.join_element(triangle);
  /// # let edges = complex.elements_of_dimension(1);
  /// # let edge = edges[0].clone();
  ///
  /// let cofaces = complex.cofaces(&edge);
  /// assert_eq!(cofaces.len(), 1); // Edge is face of triangle
  /// assert!(cofaces[0].same_content(&added_triangle));
  /// ```
  pub fn cofaces(&self, element: &T) -> Vec<T> {
    element.id().map_or_else(Vec::new, |id| {
      self
        .attachment_lattice
        .successors(id)
        .into_iter()
        .filter_map(|coface_id| self.get_element(coface_id).cloned())
        .collect()
    })
  }

  /// Computes the k-dimensional homology of the complex over a field F.
  ///
  /// Homology measures the "holes" in a topological space at different dimensions:
  /// - **H₀**: Connected components (0-dimensional holes)
  /// - **H₁**: Loops/cycles (1-dimensional holes)
  /// - **H₂**: Voids/cavities (2-dimensional holes)
  /// - **Hₖ**: k-dimensional holes (higher-dimensional features)
  ///
  /// # Mathematical Foundation
  ///
  /// Homology is defined as the quotient of cycles by boundaries:
  ///
  /// ```text
  /// Hₖ(K) = Zₖ(K) / Bₖ(K) = ker(∂ₖ) / im(∂ₖ₊₁)
  /// ```
  ///
  /// Where:
  /// - **Zₖ(K) = ker(∂ₖ)**: k-cycles (chains with no boundary)
  /// - **Bₖ(K) = im(∂ₖ₊₁)**: k-boundaries (boundaries of (k+1)-chains)
  /// - **∂ₖ**: Boundary operator from k-chains to (k-1)-chains
  ///
  /// The key insight is that "holes" are cycles that are not boundaries of higher-dimensional
  /// chains.
  ///
  /// # Algorithm
  ///
  /// 1. **Compute Cycles**: Find kernel of boundary operator ∂ₖ: Cₖ → Cₖ₋₁
  /// 2. **Compute Boundaries**: Find image of boundary operator ∂ₖ₊₁: Cₖ₊₁ → Cₖ
  /// 3. **Quotient Space**: Compute basis for quotient space Zₖ/Bₖ
  /// 4. **Return Homology**: Package result with Betti number and generators
  ///
  /// # Special Cases
  ///
  /// - **k = 0**: H₀ measures connected components. Z₀ = C₀ (all 0-chains are cycles)
  /// - **Empty Complex**: All homology groups are trivial (Hₖ = 0)
  /// - **No k-elements**: Returns trivial homology for that dimension
  ///
  /// # Field Dependency
  ///
  /// The choice of coefficient field F affects the result:
  /// - **ℤ/2ℤ (Boolean)**: Ignores orientation, counts mod 2
  /// - **ℚ (Rationals)**: Full torsion-free homology
  /// - **ℤ/pℤ (Prime fields)**: Reveals p-torsion in homology
  ///
  /// # Performance
  ///
  /// - **Time**: O(n³) where n is the number of k-dimensional elements
  /// - **Space**: O(n²) for storing boundary matrices
  /// - **Bottleneck**: Matrix kernel and image computations
  ///
  /// # Return Value
  ///
  /// Returns a [`Homology`] object containing:
  /// - `dimension`: The dimension k being computed
  /// - `betti_number`: The rank of Hₖ (number of independent k-dimensional holes)
  /// - `homology_generators`: Basis vectors representing the homology classes
  ///
  /// # Examples
  ///
  /// ## Circle (1-dimensional hole)
  ///
  /// ```rust
  /// use cova_algebra::algebras::boolean::Boolean;
  /// use cova_space::complexes::{Complex, Simplex};
  ///
  /// let mut complex = Complex::new();
  ///
  /// // Create a triangle boundary (3 edges forming a cycle)
  /// let edge1 = Simplex::new(1, vec![0, 1]);
  /// let edge2 = Simplex::new(1, vec![1, 2]);
  /// let edge3 = Simplex::new(1, vec![2, 0]);
  /// complex.join_element(edge1);
  /// complex.join_element(edge2);
  /// complex.join_element(edge3);
  ///
  /// let h0 = complex.homology::<Boolean>(0);
  /// let h1 = complex.homology::<Boolean>(1);
  ///
  /// assert_eq!(h0.betti_number, 1); // One connected component
  /// assert_eq!(h1.betti_number, 1); // One 1-dimensional hole
  /// ```
  ///
  /// ## Filled Triangle (no holes)
  ///
  /// ```rust
  /// # use cova_algebra::algebras::boolean::Boolean;
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  ///
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// complex.join_element(triangle);
  ///
  /// let h0 = complex.homology::<Boolean>(0);
  /// let h1 = complex.homology::<Boolean>(1);
  ///
  /// assert_eq!(h0.betti_number, 1); // One connected component
  /// assert_eq!(h1.betti_number, 0); // No 1D holes (filled)
  /// ```
  ///
  /// ## Sphere Surface (2-dimensional hole)
  ///
  /// ```rust
  /// # use cova_algebra::algebras::boolean::Boolean;
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  ///
  /// // Tetrahedron boundary (4 triangular faces)
  /// let face1 = Simplex::new(2, vec![0, 1, 2]);
  /// let face2 = Simplex::new(2, vec![0, 1, 3]);
  /// let face3 = Simplex::new(2, vec![0, 2, 3]);
  /// let face4 = Simplex::new(2, vec![1, 2, 3]);
  /// complex.join_element(face1);
  /// complex.join_element(face2);
  /// complex.join_element(face3);
  /// complex.join_element(face4);
  ///
  /// let h2 = complex.homology::<Boolean>(2);
  /// assert_eq!(h2.betti_number, 1); // One 2-dimensional hole (sphere interior)
  /// ```
  pub fn homology<F: Field + Copy>(&self, k: usize) -> Homology<F> {
    let k_elements = {
      let mut elements = self.elements_of_dimension(k);
      elements.sort_unstable();
      elements
    };

    if k_elements.is_empty() {
      return Homology::trivial(k);
    }

    let cycles = if k == 0 {
      // Z₀ = C₀ (kernel of ∂₀: C₀ -> C₋₁ is C₀ itself).
      let num_0_elements = k_elements.len();
      let mut basis: Vec<Vector<F>> = Vec::with_capacity(num_0_elements);
      for i in 0..num_0_elements {
        let mut v_data = vec![F::zero(); num_0_elements];
        v_data[i] = F::one();
        basis.push(Vector::new(v_data));
      }
      basis
    } else {
      let boundary_k = self.get_boundary_matrix::<F>(k);
      boundary_k.kernel()
    };

    let boundary_k_plus_1 = self.get_boundary_matrix::<F>(k + 1);
    let boundaries = boundary_k_plus_1.image();

    let quotient_basis_vectors = compute_quotient_basis(&boundaries, &cycles);

    Homology {
      dimension:           k,
      betti_number:        quotient_basis_vectors.len(),
      homology_generators: quotient_basis_vectors,
    }
  }

  /// Constructs the boundary matrix ∂ₖ: Cₖ → Cₖ₋₁ for the k-th boundary operator.
  ///
  /// The boundary matrix is the matrix representation of the linear map that takes
  /// k-dimensional chains to their (k-1)-dimensional boundaries. This is the
  /// fundamental building block for homology computations.
  ///
  /// # Mathematical Definition
  ///
  /// For a k-dimensional element σ, its boundary ∂σ is a formal sum of (k-1)-dimensional
  /// faces with appropriate orientation coefficients:
  ///
  /// ```text
  /// ∂ₖ(σ) = Σᵢ aᵢ τᵢ
  /// ```
  ///
  /// where τᵢ are the faces of σ and aᵢ ∈ F are the orientation coefficients.
  ///
  /// # Matrix Structure
  ///
  /// The resulting matrix has:
  /// - **Rows**: Indexed by (k-1)-dimensional elements (codomain basis)
  /// - **Columns**: Indexed by k-dimensional elements (domain basis)
  /// - **Entry (i,j)**: Coefficient of codomain element i in ∂(domain element j)
  ///
  /// # Element Ordering
  ///
  /// Both domain and codomain elements are sorted using their natural ordering
  /// (from [`Ord`] implementation). This ensures:
  /// - Deterministic matrix construction
  /// - Consistent results across runs
  /// - Predictable basis element correspondence
  ///
  /// # Boundary Operator Properties
  ///
  /// The matrix satisfies the fundamental property **∂² = 0**, meaning that
  /// `boundary_matrix(k+1) * boundary_matrix(k) = 0`. This is essential for
  /// the chain complex structure and homology computations.
  ///
  /// # Special Cases
  ///
  /// - **k = 0**: Returns empty matrix (0-dimensional elements have no boundary)
  /// - **No k-elements**: Returns matrix with correct row count but no columns
  /// - **No (k-1)-elements**: Returns matrix with correct column count but no rows
  ///
  /// # Implementation Details
  ///
  /// This method uses the [`ComplexElement::boundary_with_orientations`] method
  /// to get the oriented boundary of each element, then constructs the matrix
  /// by mapping faces to their positions in the codomain basis.
  ///
  /// # Performance
  ///
  /// - **Time**: O(nf) where n is the number of k-elements and f is the average number of faces
  /// - **Space**: O(nm) where m is the number of (k-1)-elements
  /// - **Optimized**: Only computes non-zero entries
  ///
  /// # Examples
  ///
  /// ## Triangle Boundary Matrix
  ///
  /// ```rust
  /// use cova_algebra::algebras::boolean::Boolean;
  /// use cova_space::complexes::{Complex, Simplex};
  ///
  /// let mut complex = Complex::new();
  /// let triangle = Simplex::new(2, vec![0, 1, 2]);
  /// complex.join_element(triangle);
  ///
  /// // Get boundary matrix ∂₂: C₂ → C₁ (triangles → edges)
  /// let boundary_2 = complex.get_boundary_matrix::<Boolean>(2);
  ///
  /// // Should be 3×1 matrix (3 edges, 1 triangle)
  /// assert_eq!(boundary_2.num_rows(), 3); // 3 edges
  /// assert_eq!(boundary_2.num_cols(), 1); // 1 triangle
  /// ```
  ///
  /// ## Edge Boundary Matrix  
  ///
  /// ```rust
  /// # use cova_algebra::algebras::boolean::Boolean;
  /// # use cova_space::complexes::{Complex, Simplex};
  /// # let mut complex = Complex::new();
  /// let edge = Simplex::new(1, vec![0, 1]);
  /// complex.join_element(edge);
  ///
  /// // Get boundary matrix ∂₁: C₁ → C₀ (edges → vertices)
  /// let boundary_1 = complex.get_boundary_matrix::<Boolean>(1);
  ///
  /// // Should be 2×1 matrix (2 vertices, 1 edge)
  /// assert_eq!(boundary_1.num_rows(), 2); // 2 vertices
  /// assert_eq!(boundary_1.num_cols(), 1); // 1 edge
  /// ```
  pub fn get_boundary_matrix<F: Field + Copy>(&self, k: usize) -> Matrix<F>
  where T: ComplexElement {
    let domain_basis = self.elements_of_dimension(k);
    let codomain_basis = self.elements_of_dimension(k.saturating_sub(1));

    if domain_basis.is_empty() || codomain_basis.is_empty() {
      // Return appropriate empty matrix
      return Matrix::zeros(codomain_basis.len(), domain_basis.len());
    }

    let mut matrix = Matrix::<F>::builder();

    // Create a map from elements to their position in the codomain basis
    let basis_map_for_codomain: HashMap<&T, usize> =
      codomain_basis.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let num_codomain_elements = codomain_basis.len();

    for element_from_domain in &domain_basis {
      // Compute boundary using the Topology trait implementation
      let boundary_chain: Chain<Self, F> = self.boundary(element_from_domain);

      // Convert the chain to a coefficient vector
      let col_vector =
        boundary_chain.to_coeff_vector(&basis_map_for_codomain, num_codomain_elements);
      matrix = matrix.column_vec(col_vector);
    }

    matrix.build()
  }
}

impl<T: ComplexElement> Default for Complex<T> {
  fn default() -> Self { Self::new() }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================
//
// The following implementations make Complex<T> integrate seamlessly with the
// broader algebraic and topological framework:
//
// - Collection: Basic containment and emptiness queries
// - Poset: Face relation partial order operations (≤, upset, downset, etc.)
// - Topology: Neighborhood and boundary operations for topological computations
//
// These implementations enable Complex<T> to work with:
// - Generic algorithms that operate on posets or topological spaces
// - Homology computations via the Topology trait boundary operator
// - Sheaf computations that require poset structure
// - General lattice-theoretic operations

/// Implementation of [`Collection`] for complexes.
///
/// Provides basic set-theoretic operations for checking element membership
/// and complex emptiness. Note that containment is based on ID equality,
/// so elements must have been added to the complex to be considered contained.
impl<T: ComplexElement> Collection for Complex<T> {
  type Item = T;

  fn contains(&self, point: &Self::Item) -> bool {
    if let Some(id) = point.id() {
      self.elements.contains_key(&id)
    } else {
      false
    }
  }

  fn is_empty(&self) -> bool { self.elements.is_empty() }
}

/// Implementation of [`Poset`] for complexes.
///
/// Defines the face relation as the partial order: σ ≤ τ means "σ is a face of τ".
/// This implementation delegates to the attachment lattice for efficiency while
/// providing the full element objects in the interface.
///
/// The face relation satisfies all poset axioms:
/// - **Reflexivity**: σ ≤ σ (every element is a face of itself)
/// - **Antisymmetry**: σ ≤ τ and τ ≤ σ implies σ = τ
/// - **Transitivity**: σ ≤ τ and τ ≤ ρ implies σ ≤ ρ
impl<T: ComplexElement> Poset for Complex<T> {
  fn leq(&self, a: &Self::Item, b: &Self::Item) -> Option<bool> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self.attachment_lattice.leq(&id_a, &id_b),
      _ => None,
    }
  }

  fn upset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .upset(id)
        .into_iter()
        .filter_map(|upset_id| self.get_element(upset_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn downset(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .downset(id)
        .into_iter()
        .filter_map(|downset_id| self.get_element(downset_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn minimal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .minimal_elements()
      .into_iter()
      .filter_map(|id| self.get_element(id).cloned())
      .collect()
  }

  fn maximal_elements(&self) -> std::collections::HashSet<Self::Item> {
    self
      .attachment_lattice
      .maximal_elements()
      .into_iter()
      .filter_map(|id| self.get_element(id).cloned())
      .collect()
  }

  fn join(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self
        .attachment_lattice
        .join(id_a, id_b)
        .and_then(|join_id| self.get_element(join_id).cloned()),
      _ => None,
    }
  }

  fn meet(&self, a: Self::Item, b: Self::Item) -> Option<Self::Item> {
    match (a.id(), b.id()) {
      (Some(id_a), Some(id_b)) => self
        .attachment_lattice
        .meet(id_a, id_b)
        .and_then(|meet_id| self.get_element(meet_id).cloned()),
      _ => None,
    }
  }

  fn successors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .successors(id)
        .into_iter()
        .filter_map(|succ_id| self.get_element(succ_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }

  fn predecessors(&self, a: Self::Item) -> std::collections::HashSet<Self::Item> {
    if let Some(id) = a.id() {
      self
        .attachment_lattice
        .predecessors(id)
        .into_iter()
        .filter_map(|pred_id| self.get_element(pred_id).cloned())
        .collect()
    } else {
      std::collections::HashSet::new()
    }
  }
}

/// Implementation of [`Topology`] for complexes.
///
/// Provides topological operations that integrate with the broader framework:
/// - **Neighborhood**: Returns cofaces (elements containing the given element)
/// - **Boundary**: Computes oriented boundary using element-specific operators
///
/// The boundary implementation is crucial for homology computations as it provides
/// the chain complex structure with proper orientation handling.
impl<T: ComplexElement> Topology for Complex<T> {
  fn neighborhood(&self, item: &Self::Item) -> Vec<Self::Item> {
    // Return direct cofaces (elements that have this item as a face)
    self.cofaces(item)
  }

  fn boundary<R: Ring + Copy>(&self, item: &Self::Item) -> Chain<Self, R> {
    if item.dimension() == 0 {
      return Chain::new(self);
    }

    let mut boundary_chain_items = Vec::new();
    let mut boundary_chain_coeffs = Vec::new();

    // Use the element-specific boundary computation with orientations
    let faces_with_orientations = item.boundary_with_orientations();

    for (face, orientation) in faces_with_orientations {
      // Find the corresponding element in the complex that matches this face's content
      if let Some(complex_face) = self.find_equivalent_element(&face) {
        // Use the orientation coefficient from the element-specific boundary operator
        let coeff = if orientation > 0 {
          R::one()
        } else if orientation < 0 {
          -R::one()
        } else {
          continue; // Skip faces with zero coefficient
        };
        boundary_chain_items.push(complex_face);
        boundary_chain_coeffs.push(coeff);
      }
    }

    Chain::from_items_and_coeffs(self, boundary_chain_items, boundary_chain_coeffs)
  }
}

#[cfg(test)]
mod tests {
  use cova_algebra::algebras::boolean::Boolean;

  use super::*;

  // =============================================================================
  // COMPREHENSIVE TEST SUITE
  // =============================================================================
  //
  // This test suite demonstrates and validates the key features of the complexes
  // module across multiple dimensions:
  //
  // 1. **Generic Complex Operations**: Tests that work with both simplicial and cubical complexes,
  //    showing the power of the generic Complex<T> design
  //
  // 2. **Closure Property**: Verifies that adding elements automatically includes all faces,
  //    maintaining the fundamental complex property
  //
  // 3. **ID Management**: Tests automatic ID assignment, deduplication, and proper handling of
  //    elements with/without IDs
  //
  // 4. **Poset Structure**: Validates that face relations form a proper partial order with correct
  //    upset/downset computations
  //
  // 5. **Topology Integration**: Tests neighborhood operations and boundary computations that
  //    integrate with the topology framework
  //
  // 6. **Homology Computations**: Comprehensive tests of homology computation for standard
  //    topological spaces (circles, spheres, etc.) over different coefficient fields
  //
  // 7. **Cross-Complex Compatibility**: Demonstrates that the same algorithms work on both
  //    simplicial and cubical complexes, producing mathematically consistent results
  //
  // These tests serve both as validation and as examples of proper usage patterns.

  #[test]
  fn test_generic_complex_with_simplex() {
    let mut complex: Complex<Simplex> = Complex::new();

    // Create a triangle (2-simplex)
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    // Should have 1 triangle, 3 edges, 3 vertices
    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert_eq!(complex.elements_of_dimension(1).len(), 3);
    assert_eq!(complex.elements_of_dimension(0).len(), 3);

    // Check that the triangle is in the complex (use the returned element with ID)
    assert!(complex.contains(&added_triangle));

    // Check lattice relationships
    let faces = complex.faces(&added_triangle);
    assert_eq!(faces.len(), 3); // Triangle should have 3 edges as direct faces
  }

  #[test]
  fn test_generic_complex_with_cube() {
    let mut complex: Complex<Cube> = Complex::new();

    // Create a square (2-cube)
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = complex.join_element(square);

    // Should have 1 square, 4 edges, 4 vertices
    assert_eq!(complex.elements_of_dimension(2).len(), 1);
    assert_eq!(complex.elements_of_dimension(1).len(), 4);
    assert_eq!(complex.elements_of_dimension(0).len(), 4);

    // Check that the square is in the complex (use the returned element with ID)
    assert!(complex.contains(&added_square));

    // Check lattice relationships
    let faces = complex.faces(&added_square);
    assert_eq!(faces.len(), 4); // Square should have 4 edges as direct faces
  }

  #[test]
  fn test_automatic_id_assignment() {
    let mut complex = SimplicialComplex::new();

    // Create elements without IDs
    let vertex1 = Simplex::new(0, vec![0]);
    let vertex2 = Simplex::new(0, vec![1]);
    let edge = Simplex::new(1, vec![0, 1]);

    // Add them to the complex - should get automatic ID assignment
    let added_vertex1 = complex.join_element(vertex1);
    let added_vertex2 = complex.join_element(vertex2);
    let added_edge = complex.join_element(edge);

    // All should be contained
    assert!(complex.contains(&added_vertex1));
    assert!(complex.contains(&added_vertex2));
    assert!(complex.contains(&added_edge));

    // Should have assigned IDs
    assert!(added_vertex1.id().is_some());
    assert!(added_vertex2.id().is_some());
    assert!(added_edge.id().is_some());

    // Should have proper lattice relationships
    assert!(complex.leq(&added_vertex1, &added_edge).unwrap());
    assert!(complex.leq(&added_vertex2, &added_edge).unwrap());
  }

  #[test]
  fn test_element_deduplication() {
    let mut complex = SimplicialComplex::new();

    // Create two mathematically identical simplices
    let simplex1 = Simplex::new(1, vec![0, 1]);
    let simplex2 = Simplex::new(1, vec![0, 1]); // Same content

    let added1 = complex.join_element(simplex1);
    let added2 = complex.join_element(simplex2);

    // Should return the same element (by ID)
    assert_eq!(added1.id(), added2.id());
    assert_eq!(complex.elements_of_dimension(1).len(), 1); // Only one edge stored
  }

  // =====================================================
  // GENERIC POSET OPERATIONS TESTS
  // =====================================================

  #[test]
  fn test_complex_poset_operations() {
    let mut complex = SimplicialComplex::new();

    let triangle = Simplex::new(2, vec![0, 1, 2]);
    let added_triangle = complex.join_element(triangle);

    // Get the actual elements with IDs from the complex
    let vertices = complex.elements_of_dimension(0);
    let edges = complex.elements_of_dimension(1);

    // Find specific vertex and edge by content
    let vertex = vertices.iter().find(|v| v.vertices() == [0]).unwrap().clone();
    let edge = edges.iter().find(|e| e.vertices() == [0, 1]).unwrap().clone();

    // Test leq relationships
    assert_eq!(complex.leq(&vertex, &edge), Some(true));
    assert_eq!(complex.leq(&edge, &added_triangle), Some(true));
    assert_eq!(complex.leq(&vertex, &added_triangle), Some(true));
    assert_eq!(complex.leq(&added_triangle, &vertex), Some(false));

    // Test upset/downset
    let vertex_upset = complex.upset(vertex.clone());
    assert!(vertex_upset.contains(&vertex));
    assert!(vertex_upset.contains(&edge));
    assert!(vertex_upset.contains(&added_triangle));

    let triangle_downset = complex.downset(added_triangle.clone());
    assert!(triangle_downset.contains(&vertex));
    assert!(triangle_downset.contains(&edge));
    assert!(triangle_downset.contains(&added_triangle));
  }

  #[test]
  fn test_complex_topology_operations() {
    let mut complex = SimplicialComplex::new();

    let edge = Simplex::new(1, vec![0, 1]);
    let added_edge = complex.join_element(edge);

    // Get the actual vertex with ID from the complex
    let vertices = complex.elements_of_dimension(0);
    let vertex = vertices.iter().find(|v| v.vertices() == [0]).unwrap().clone();

    // Test neighborhood (cofaces)
    let vertex_neighborhood = complex.neighborhood(&vertex);
    assert_eq!(vertex_neighborhood.len(), 1);
    assert!(vertex_neighborhood.contains(&added_edge));

    let edge_neighborhood = complex.neighborhood(&added_edge);
    assert_eq!(edge_neighborhood.len(), 0); // No 2-simplices attached
  }

  #[test]
  fn test_mixed_complex_operations() {
    // Test that we can use the same generic operations on both simplicial and cubical complexes

    // Simplicial complex
    let mut simplicial_complex = SimplicialComplex::new();
    let triangle = Simplex::from_vertices(vec![0, 1, 2]);
    let added_triangle = simplicial_complex.join_element(triangle);

    // Should automatically add all faces
    assert_eq!(simplicial_complex.elements_of_dimension(2).len(), 1);
    assert_eq!(simplicial_complex.elements_of_dimension(1).len(), 3);
    assert_eq!(simplicial_complex.elements_of_dimension(0).len(), 3);

    // Cubical complex
    let mut cubical_complex = CubicalComplex::new();
    let square = Cube::square([0, 1, 2, 3]);
    let added_square = cubical_complex.join_element(square);

    // Should automatically add all faces
    assert_eq!(cubical_complex.elements_of_dimension(2).len(), 1);
    assert_eq!(cubical_complex.elements_of_dimension(1).len(), 4);
    assert_eq!(cubical_complex.elements_of_dimension(0).len(), 4);

    // Both should support the same interface
    assert!(simplicial_complex.contains(&added_triangle));
    assert!(cubical_complex.contains(&added_square));

    // Both should have proper lattice structure
    let triangle_faces = simplicial_complex.faces(&added_triangle);
    let square_faces = cubical_complex.faces(&added_square);
    assert_eq!(triangle_faces.len(), 3); // triangle has 3 edges
    assert_eq!(square_faces.len(), 4); // square has 4 edges
  }

  #[test]
  fn test_generic_homology_computation() {
    // Test simplicial complex - triangle boundary (should have H_1 = 1)
    let mut simplicial_complex = SimplicialComplex::new();

    // Create a triangle boundary (3 edges forming a cycle)
    let edge1 = Simplex::new(1, vec![0, 1]);
    let edge2 = Simplex::new(1, vec![1, 2]);
    let edge3 = Simplex::new(1, vec![2, 0]);

    simplicial_complex.join_element(edge1);
    simplicial_complex.join_element(edge2);
    simplicial_complex.join_element(edge3);

    // Compute homology over Z/2Z
    let h0 = simplicial_complex.homology::<Boolean>(0);
    let h1 = simplicial_complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 1); // One 1-dimensional hole

    // Test cubical complex - square boundary (should also have H_1 = 1)
    let mut cubical_complex = CubicalComplex::new();

    // Create a square boundary (4 edges forming a cycle)
    let edge1 = Cube::edge(0, 1);
    let edge2 = Cube::edge(1, 2);
    let edge3 = Cube::edge(2, 3);
    let edge4 = Cube::edge(3, 0);

    cubical_complex.join_element(edge1);
    cubical_complex.join_element(edge2);
    cubical_complex.join_element(edge3);
    cubical_complex.join_element(edge4);

    // Compute homology over Z/2Z
    let h0_cube = cubical_complex.homology::<Boolean>(0);
    let h1_cube = cubical_complex.homology::<Boolean>(1);

    assert_eq!(h0_cube.betti_number, 1); // One connected component
    assert_eq!(h1_cube.betti_number, 1); // One 1-dimensional hole

    // Both simplicial and cubical complexes should give same topological result
    assert_eq!(h0.betti_number, h0_cube.betti_number);
    assert_eq!(h1.betti_number, h1_cube.betti_number);
  }

  #[test]
  fn test_generic_homology_filled_shapes() {
    // Test filled triangle (should have H_1 = 0)
    let mut simplicial_complex = SimplicialComplex::new();
    let triangle = Simplex::new(2, vec![0, 1, 2]);
    simplicial_complex.join_element(triangle);

    let h0 = simplicial_complex.homology::<Boolean>(0);
    let h1 = simplicial_complex.homology::<Boolean>(1);

    assert_eq!(h0.betti_number, 1); // One connected component
    assert_eq!(h1.betti_number, 0); // No 1D holes (filled)

    // Test filled square (should also have H_1 = 0)
    let mut cubical_complex = CubicalComplex::new();
    let square = Cube::square([0, 1, 2, 3]);
    cubical_complex.join_element(square);

    let h0_cube = cubical_complex.homology::<Boolean>(0);
    let h1_cube = cubical_complex.homology::<Boolean>(1);

    assert_eq!(h0_cube.betti_number, 1); // One connected component
    assert_eq!(h1_cube.betti_number, 0); // No 1D holes (filled)
  }
}
