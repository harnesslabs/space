//! # Category Theory Primitives
//!
//! This module provides a basic trait, [`Category`], for representing abstract
//! categories in mathematics. A category consists of objects and morphisms
//! (or arrows) between these objects.
//!
//! ## Mathematical Definition
//!
//! A category $\mathcal{C}$ consists of:
//! - A collection of **objects**, denoted $\text{ob}(\mathcal{C})$.
//! - For every pair of objects $A, B \in \text{ob}(\mathcal{C})$, a collection of **morphisms** (or
//!   arrows) from $A$ to $B$, denoted $\text{Hom}_{\mathcal{C}}(A, B)$. If $f \in
//!   \text{Hom}_{\mathcal{C}}(A, B)$, we write $f: A \to B$.
//! - For every object $A \in \text{ob}(\mathcal{C})$, an **identity morphism** $\text{id}_A: A \to
//!   A$.
//! - For every triple of objects $A, B, C \in \text{ob}(\mathcal{C})$, a binary operation called
//!   **composition of morphisms**, $ \circ : \text{Hom}_{\mathcal{C}}(B, C) \times
//!   \text{Hom}_{\mathcal{C}}(A, B) \to \text{Hom}_{\mathcal{C}}(A, C)$. Given $g: B \to C$ and $f:
//!   A \to B$, their composition is written $g \circ f: A \to C$.
//!
//! These components must satisfy two axioms:
//! 1. **Associativity**: For any morphisms $f: A \to B$, $g: B \to C$, and $h: C \to D$, the
//!    equation $h \circ (g \circ f) = (h \circ g) \circ f$ must hold.
//! 2. **Identity**: For any morphism $f: A \to B$, the equations $\text{id}_B \circ f = f$ and $f
//!    \circ \text{id}_A = f$ must hold.
//!
//! In this module, the objects of the category are represented by types that implement
//! the [`Category`] trait itself. The morphisms are represented by an associated type
//! [`Category::Morphism`].

// TODO (autoparallel): It may be smarter to have these use references instead of
// ownership. This way we can avoid unnecessary cloning.
/// Represents an object in a category, along with its morphisms and operations.
///
/// In category theory, a category consists of objects and morphisms between them.
/// This trait models an object within such a category. The type implementing `Category`
/// acts as an object, and it defines an associated type `Morphism` for the arrows.
///
/// The trait provides methods for morphism composition, obtaining identity morphisms,
/// and applying a morphism to an object (which can be thought of as evaluating a function
/// if objects are sets and morphisms are functions, or as a more abstract action).
pub trait Category: Sized {
  /// The type of morphisms (arrows) between objects in this category.
  /// For example, if objects are sets, morphisms could be functions.
  /// If objects are vector spaces, morphisms could be linear maps.
  type Morphism;

  /// Composes two morphisms `f` and `g`.
  ///
  /// If $g: B \to C$ and $f: A \to B$, then `compose(g, f)` results in a morphism $g \circ f: A \to
  /// C$. Note the order: `g` is applied after `f`.
  ///
  /// # Arguments
  ///
  /// * `f`: The first morphism to apply (e.g., $f: A \to B$).
  /// * `g`: The second morphism to apply (e.g., $g: B \to C$).
  ///
  /// # Returns
  ///
  /// The composed morphism $g \circ f$.
  fn compose(f: Self::Morphism, g: Self::Morphism) -> Self::Morphism;

  /// Returns the identity morphism for a given object `a`.
  ///
  /// The identity morphism $\text{id}_a: a \to a$ is such that for any morphism
  /// $f: X \to a$, $\text{id}_a \circ f = f$, and for any morphism $g: a \to Y$,
  /// $g \circ \text{id}_a = g$.
  ///
  /// # Arguments
  ///
  /// * `a`: The object for which to get the identity morphism. In this trait, `Self` is the object.
  ///
  /// # Returns
  ///
  /// The identity morphism for object `a`.
  fn identity(a: Self) -> Self::Morphism;

  /// Applies a morphism `f` to an object `x`.
  ///
  /// This can be interpreted in various ways depending on the specific category.
  /// If objects are sets and `x` is an element of an object `A` (represented by `Self`),
  /// and $f: A \to B$ is a morphism, then `apply(f, x)` could represent $f(x)$, an element of `B`.
  /// More generally, if `Self` represents an object `A` and `f` is a morphism $A \to B$,
  /// this function should return an object of type `Self` representing `B` after the action of `f`
  /// on `A` (or an element of `B` if `Self` represents elements).
  ///
  /// The exact semantics (whether `Self` represents an object or an element of an object,
  /// and what `apply` means) might need clarification based on usage.
  /// Given the return type is `Self`, it suggests `f` might be an endomorphism ($f: A \to A$)
  /// and `x` is an instance of `A`, or that `Self` represents elements and `f` maps elements of
  /// one object (type) to elements of another (potentially same type `Self`).
  ///
  /// # Arguments
  ///
  /// * `f`: The morphism to apply.
  /// * `x`: The object (or element of an object) to which the morphism is applied.
  ///
  /// # Returns
  ///
  /// The result of applying morphism `f` to `x`.
  fn apply(f: Self::Morphism, x: Self) -> Self;
}
