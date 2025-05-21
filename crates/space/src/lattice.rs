//! # Lattice Module
//! Defines a generic lattice data structure and associated operations.
//! This module provides `Lattice<T>`, a structure capable of representing
//! any partially ordered set where `T` is the type of elements in the lattice.
//!
//! ## Features
//! - Creation of lattices and addition of elements and relations (`a ≤ b`).
//! - Automatic computation of transitive closure to represent all implied relations.
//! - Checking for relations (`leq`).
//! - Finding minimal and maximal elements.
//! - Computation of join (least upper bound) and meet (greatest lower bound) of elements.
//! - Exporting the lattice structure to a DOT file format for visualization (e.g., with Graphviz).
//!
//! ## Type Constraints
//! - The element type `T` must implement `std::hash::Hash`, `std::cmp::Eq`, and `std::clone::Clone`
//!   for basic lattice operations.
//! - For generating DOT file output using `save_to_dot_file`, `T` must also implement
//!   `std::fmt::Display` (for node labels) and `std::cmp::Ord` (for consistent output ordering).
//!
//! ## Example Usage
//! ```
//! use harness_space::{lattice::Lattice, prelude::*};
//!
//! // Create a new lattice for integers
//! let mut lattice: Lattice<i32> = Lattice::new();
//!
//! // Add elements and relations (e.g., a simple chain 1 ≤ 2 ≤ 3)
//! lattice.add_relation(1, 2);
//! lattice.add_relation(2, 3);
//!
//! // Check relations
//! assert!(lattice.leq(&1, &3).unwrap());
//! assert!(!lattice.leq(&3, &1).unwrap());
//!
//! // Find minimal and maximal elements
//! let minimal = lattice.minimal_elements();
//! assert!(minimal.contains(&1) && minimal.len() == 1);
//! let maximal = lattice.maximal_elements();
//! assert!(maximal.contains(&3) && maximal.len() == 1);
//!
//! // Compute join and meet (for a diamond lattice, for example)
//! let mut diamond: Lattice<char> = Lattice::new();
//! diamond.add_relation('d', 'b'); // d is bottom
//! diamond.add_relation('d', 'c');
//! diamond.add_relation('b', 'a'); // a is top
//! diamond.add_relation('c', 'a');
//!
//! assert_eq!(diamond.join('b', 'c'), Some('a'));
//! assert_eq!(diamond.meet('b', 'c'), Some('d'));
//!
//! // Save to a DOT file (requires T: Display + Ord)
//! // if let Err(e) = diamond.save_to_dot_file("diamond_lattice.dot") {
//! //     eprintln!("Failed to save: {}", e);
//! // }
//! ```

use std::{
  collections::{HashMap, HashSet},
  fs::File,
  hash::Hash,
  io::{Result as IoResult, Write as IoWrite},
};

use crate::set::{Collection, Poset};

/// A node in a lattice representing an element and its relationships.
///
/// Each node stores an element of type `T` and maintains sets of its direct
/// successors (elements greater than this one) and direct predecessors
/// (elements less than this one) in the partial order.
#[derive(Debug, Clone)]
pub struct LatticeNode<T> {
  /// The element stored in this node.
  element:      T,
  /// Direct successors (elements that are greater than this one according to the
  /// lattice's partial order).
  successors:   HashSet<T>,
  /// Direct predecessors (elements that are less than this one according to the
  /// lattice's partial order).
  predecessors: HashSet<T>,
}

/// A general lattice structure that can represent any partially ordered set
/// with join and meet operations.
///
/// A lattice is a partially ordered set in which any two elements have a unique
/// supremum (also called a least upper bound or join) and a unique infimum
/// (also called a greatest lower bound or meet).
/// This implementation stores elements of type `T` and their relationships.
/// The type `T` must implement `Hash`, `Eq`, and `Clone`.
/// For DOT file generation, `T` must also implement `Display` and `Ord`.
#[derive(Debug, Default, Clone)]
pub struct Lattice<T> {
  /// Map of elements to their nodes. Each key is an element in the lattice,
  /// and its value is the `LatticeNode` containing the element's relationships.
  nodes: HashMap<T, LatticeNode<T>>,
}

impl<T: Hash + Eq + Clone> Lattice<T> {
  /// Creates a new empty lattice.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let lattice: Lattice<i32> = Lattice::new();
  /// ```
  pub fn new() -> Self { Self { nodes: HashMap::new() } }

  /// Adds a new element to the lattice.
  ///
  /// If the element already exists in the lattice, this method does nothing.
  ///
  /// # Arguments
  ///
  /// * `element`: The element to add to the lattice.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new();
  /// lattice.add_element(1);
  /// ```
  pub fn add_element(&mut self, element: T) {
    if !self.nodes.contains_key(&element) {
      self.nodes.insert(element.clone(), LatticeNode {
        element,
        successors: HashSet::new(),
        predecessors: HashSet::new(),
      });
    }
  }

  /// Adds a relation `a ≤ b` to the lattice.
  ///
  /// This indicates that element `a` is less than or equal to element `b`
  /// in the partial order. If `a` or `b` are not already in the lattice,
  /// they are added.
  ///
  /// After adding the direct relation, this method also updates the transitive
  /// closure of the lattice to ensure all indirect relationships are captured.
  ///
  /// # Arguments
  ///
  /// * `a`: The smaller element in the relation.
  /// * `b`: The greater element in the relation.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new();
  /// lattice.add_relation(1, 2); // 1 ≤ 2
  /// ```
  pub fn add_relation(&mut self, a: T, b: T) {
    self.add_element(a.clone());
    self.add_element(b.clone());

    // Add direct relation
    if let Some(node_a) = self.nodes.get_mut(&a) {
      node_a.successors.insert(b.clone());
    }
    if let Some(node_b) = self.nodes.get_mut(&b) {
      node_b.predecessors.insert(a);
    }

    // Transitive closure
    self.compute_transitive_closure();
  }

  /// Computes the transitive closure of the lattice
  fn compute_transitive_closure(&mut self) {
    let mut changed = true;
    while changed {
      changed = false;
      let mut updates = Vec::new();

      // Collect all updates first
      for node in self.nodes.values() {
        for succ in &node.successors {
          if let Some(succ_node) = self.nodes.get(succ) {
            for succ_succ in &succ_node.successors {
              updates.push((node.element.clone(), succ_succ.clone()));
            }
          }
        }
      }

      // Apply updates
      for (a, b) in updates {
        if let Some(node_a) = self.nodes.get_mut(&a) {
          if node_a.successors.insert(b.clone()) {
            changed = true;
          }
        }
        if let Some(node_b) = self.nodes.get_mut(&b) {
          if node_b.predecessors.insert(a) {
            changed = true;
          }
        }
      }
    }
  }
}

impl<T: Hash + Eq + Clone> Collection for Lattice<T> {
  type Item = T;

  fn contains(&self, point: &Self::Item) -> bool { self.nodes.contains_key(point) }

  fn is_empty(&self) -> bool { self.nodes.is_empty() }
}

impl<T: Hash + Eq + Clone> Poset for Lattice<T> {
  /// Checks if `a ≤ b` in the lattice.
  ///
  /// This method determines if element `a` is less than or equal to element `b`
  /// according to the partial order defined in the lattice. This relies on the
  /// precomputed transitive closure.
  ///
  /// # Arguments
  ///
  /// * `a`: The first element.
  /// * `b`: The second element.
  ///
  /// # Returns
  ///
  /// `true` if `a ≤ b`, `false` otherwise. Returns `false` if either `a` or `b`
  /// is not in the lattice.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new();
  /// lattice.add_relation(1, 2);
  /// lattice.add_relation(2, 3);
  /// assert!(lattice.leq(&1, &3).unwrap()); // Transitive: 1 ≤ 2 and 2 ≤ 3 => 1 ≤ 3
  /// assert!(!lattice.leq(&3, &1).unwrap());
  /// ```
  fn leq(&self, a: &T, b: &T) -> Option<bool> {
    if !self.nodes.contains_key(a) || !self.nodes.contains_key(b) {
      return None;
    }
    if a == b {
      return Some(true);
    }
    let node_a = self.nodes.get(a).unwrap();
    Some(node_a.successors.contains(b))
  }

  /// Returns all minimal elements in the lattice.
  ///
  /// Minimal elements are those that have no predecessors in the partial order.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all minimal elements of the lattice.
  /// If the lattice is empty, an empty set is returned.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new();
  /// lattice.add_relation(1, 2);
  /// lattice.add_relation(1, 3);
  /// let minimal = lattice.minimal_elements();
  /// assert!(minimal.contains(&1) && minimal.len() == 1);
  /// ```
  fn minimal_elements(&self) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| node.predecessors.is_empty())
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Returns all maximal elements in the lattice.
  ///
  /// Maximal elements are those that have no successors in the partial order.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all maximal elements of the lattice.
  /// If the lattice is empty, an empty set is returned.
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new();
  /// lattice.add_relation(1, 3);
  /// lattice.add_relation(2, 3);
  /// let maximal = lattice.maximal_elements();
  /// assert!(maximal.contains(&3) && maximal.len() == 1);
  /// ```
  fn maximal_elements(&self) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| node.successors.is_empty())
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Computes the join (least upper bound) of two elements `a` and `b`.
  ///
  /// The join of `a` and `b` is an element `x` such that `a ≤ x` and `b ≤ x`,
  /// and for any other element `y` with `a ≤ y` and `b ≤ y`, it holds that `x ≤ y`.
  ///
  /// # Arguments
  ///
  /// * `a`: The first element.
  /// * `b`: The second element.
  ///
  /// # Returns
  ///
  /// An `Option<T>` containing the unique join of `a` and `b` if it exists.
  /// Returns `None` if:
  /// * Either `a` or `b` is not in the lattice.
  /// * `a` and `b` have no common upper bounds.
  /// * `a` and `b` have multiple minimal common upper bounds (i.e., the join is not unique).
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new(); // Diamond lattice
  /// lattice.add_relation(4, 2);
  /// lattice.add_relation(4, 3);
  /// lattice.add_relation(2, 1);
  /// lattice.add_relation(3, 1);
  /// // Here, 4 is bottom, 1 is top.
  /// assert_eq!(lattice.join(2, 3), Some(1));
  /// assert_eq!(lattice.join(4, 2), Some(2));
  /// ```
  fn join(&self, a: T, b: T) -> Option<T> {
    if !self.nodes.contains_key(&a) || !self.nodes.contains_key(&b) {
      return None; // Elements must be in the lattice
    }

    let node_a = self.nodes.get(&a).unwrap();
    let node_b = self.nodes.get(&b).unwrap();

    let mut upper_bounds_a = node_a.successors.iter().cloned().collect::<HashSet<T>>();
    upper_bounds_a.insert(a.clone());

    let mut upper_bounds_b = node_b.successors.iter().cloned().collect::<HashSet<T>>();
    upper_bounds_b.insert(b.clone());

    let common_upper_bounds: HashSet<T> =
      upper_bounds_a.intersection(&upper_bounds_b).cloned().collect();

    if common_upper_bounds.is_empty() {
      return None;
    }

    let minimal_common_upper_bounds: Vec<T> = common_upper_bounds
      .iter()
      .filter(|&x| common_upper_bounds.iter().all(|y| x == y || !self.leq(y, x).unwrap_or(false)))
      .cloned()
      .collect();

    if minimal_common_upper_bounds.len() == 1 {
      Some(minimal_common_upper_bounds[0].clone())
    } else {
      None
    }
  }

  /// Computes the meet (greatest lower bound) of two elements `a` and `b`.
  ///
  /// The meet of `a` and `b` is an element `x` such that `x ≤ a` and `x ≤ b`,
  /// and for any other element `y` with `y ≤ a` and `y ≤ b`, it holds that `y ≤ x`.
  ///
  /// # Arguments
  ///
  /// * `a`: The first element.
  /// * `b`: The second element.
  ///
  /// # Returns
  ///
  /// An `Option<T>` containing the unique meet of `a` and `b` if it exists.
  /// Returns `None` if:
  /// * Either `a` or `b` is not in the lattice.
  /// * `a` and `b` have no common lower bounds.
  /// * `a` and `b` have multiple maximal common lower bounds (i.e., the meet is not unique).
  ///
  /// # Examples
  ///
  /// ```
  /// use harness_space::{lattice::Lattice, prelude::*};
  /// let mut lattice = Lattice::new(); // Diamond lattice
  /// lattice.add_relation(4, 2);
  /// lattice.add_relation(4, 3);
  /// lattice.add_relation(2, 1);
  /// lattice.add_relation(3, 1);
  /// // Here, 4 is bottom, 1 is top.
  /// assert_eq!(lattice.meet(2, 3), Some(4));
  /// assert_eq!(lattice.meet(1, 2), Some(2));
  /// ```
  fn meet(&self, a: T, b: T) -> Option<T> {
    if !self.nodes.contains_key(&a) || !self.nodes.contains_key(&b) {
      return None; // Elements must be in the lattice
    }

    let node_a = self.nodes.get(&a).unwrap();
    let node_b = self.nodes.get(&b).unwrap();

    let mut lower_bounds_a = node_a.predecessors.iter().cloned().collect::<HashSet<T>>();
    lower_bounds_a.insert(a.clone());

    let mut lower_bounds_b = node_b.predecessors.iter().cloned().collect::<HashSet<T>>();
    lower_bounds_b.insert(b.clone());

    let common_lower_bounds: HashSet<T> =
      lower_bounds_a.intersection(&lower_bounds_b).cloned().collect();

    if common_lower_bounds.is_empty() {
      return None;
    }

    let maximal_common_lower_bounds: Vec<T> = common_lower_bounds
      .iter()
      .filter(|&x| common_lower_bounds.iter().all(|y| x == y || !self.leq(x, y).unwrap_or(false)))
      .cloned()
      .collect();

    if maximal_common_lower_bounds.len() == 1 {
      Some(maximal_common_lower_bounds[0].clone())
    } else {
      None
    }
  }

  /// Returns the set of elements that are less than or equal to `a`.
  ///
  /// This method returns a `HashSet` containing all elements in the lattice
  /// that are less than or equal to `a`. If `a` is not in the lattice,
  /// the method returns an empty set.
  ///
  /// # Arguments
  ///
  /// * `a`: The element to get the lower bounds of.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all elements in the lattice that are less than or equal to `a`.
  fn downset(&self, a: T) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| self.leq(&node.element, &a).unwrap_or(false))
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Returns the set of elements that are greater than or equal to `a`.
  ///
  /// This method returns a `HashSet` containing all elements in the lattice
  /// that are greater than or equal to `a`. If `a` is not in the lattice,
  /// the method returns an empty set.
  ///
  /// # Arguments
  ///
  /// * `a`: The element to get the upper bounds of.
  ///
  /// # Returns
  ///
  /// A `HashSet` containing all elements in the lattice that are greater than or equal to `a`.
  fn upset(&self, a: T) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| self.leq(&a, &node.element).unwrap_or(false))
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Returns the set of elements that are successors of `a`.
  ///
  /// This method returns a `HashSet` containing all elements in the lattice
  /// that are direct successors of `a`. If `a` is not in the lattice,
  /// the method returns an empty set.
  fn successors(&self, a: T) -> HashSet<T> {
    self.nodes.get(&a).map_or_else(HashSet::new, |node_a| {
      // Filter to only include direct successors
      let all_successors = &node_a.successors;
      all_successors
        .iter()
        .filter(|&b| {
          // A successor b is direct if there's no other element c where a < c < b
          !all_successors.iter().any(|c| {
            c != b && self.nodes.get(c).is_some_and(|node_c| node_c.successors.contains(b))
          })
        })
        .cloned()
        .collect()
    })
  }

  /// Returns the set of elements that are predecessors of `a`.
  ///
  /// This method returns a `HashSet` containing all elements in the lattice
  /// that are direct predecessors of `a`. If `a` is not in the lattice,
  /// the method returns an empty set.
  fn predecessors(&self, a: T) -> HashSet<T> {
    self.nodes.get(&a).map_or_else(HashSet::new, |node_a| {
      // Filter to only include direct predecessors
      let all_predecessors = &node_a.predecessors;
      all_predecessors
        .iter()
        .filter(|&b| {
          // A predecessor b is direct if there's no other element c where b < c < a
          !all_predecessors.iter().any(|c| {
            c != b && self.nodes.get(c).is_some_and(|node_c| node_c.predecessors.contains(b))
          })
        })
        .cloned()
        .collect()
    })
  }
}

// Helper function to escape strings for DOT format
fn escape_dot_label(label: &str) -> String { label.replace('"', "\\\"") }

// Implementation block for methods requiring Display and Ord for T
impl<T: Hash + Eq + Clone + std::fmt::Display + Ord> Lattice<T> {
  /// Saves the lattice representation in DOT format to the specified file.
  ///
  /// This method generates a string in the DOT language (used by Graphviz)
  /// representing the Hasse diagram of the lattice and writes it to the given file.
  /// The diagram shows only the covering relations (immediate successor/predecessor).
  /// Elements are displayed using their `Display` implementation.
  /// The layout is bottom-to-top (`rankdir="BT"`).
  ///
  /// Requires `T` to implement `std::fmt::Display` and `std::cmp::Ord`
  /// for consistent node labeling and ordering in the output.
  ///
  /// # Arguments
  ///
  /// * `filename`: The path to the file where the DOT representation will be saved. If the file
  ///   exists, it will be overwritten.
  ///
  /// # Returns
  ///
  /// An `IoResult<()>` which is `Ok(())` on successful write, or an `Err`
  /// containing an `std::io::Error` if any I/O error occurs (e.g., file
  /// creation fails).
  ///
  /// # Panics
  ///
  /// This method does not explicitly panic, but file operations can panic
  /// under certain unrecoverable conditions (though `File::create` and `writeln!`
  /// typically return `Result`).
  ///
  /// # Examples
  ///
  /// ```no_run
  /// use harness_space::lattice::Lattice;
  /// let mut lattice = Lattice::new();
  /// lattice.add_relation("a", "b");
  /// lattice.add_relation("b", "c");
  /// if let Err(e) = lattice.save_to_dot_file("my_lattice.dot") {
  ///   eprintln!("Failed to save lattice: {}", e);
  /// }
  /// ```
  ///
  /// The resulting `my_lattice.dot` file would look something like:
  /// ```dot
  /// digraph Lattice {
  ///   rankdir="BT";
  ///   node [shape=plaintext];
  ///   "a";
  ///   "b";
  ///   "c";
  ///
  ///   "a" -> "b";
  ///   "b" -> "c";
  /// }
  /// ```
  pub fn save_to_dot_file(&self, filename: &str) -> IoResult<()> {
    let mut file = File::create(filename)?;

    if self.nodes.is_empty() {
      return writeln!(file, "digraph Lattice {{\n  label=\"Empty Lattice\";\n}}");
    }

    writeln!(file, "digraph Lattice {{")?;
    writeln!(file, "  rankdir=\"BT\";")?;
    writeln!(file, "  node [shape=plaintext];")?;

    let mut sorted_node_keys: Vec<&T> = self.nodes.keys().collect();
    // T: Ord is required for sorted_node_keys.sort()
    // &T will be sorted based on the Ord impl of T.
    sorted_node_keys.sort();

    // Define nodes
    for node_key_ptr in &sorted_node_keys {
      let node_key = *node_key_ptr; // node_key is &T
                                    // node_key.to_string() requires T: std::fmt::Display
      writeln!(file, "  \"{}\";", escape_dot_label(&node_key.to_string()))?;
    }
    writeln!(file)?; // Blank line for readability

    // Define edges (covering relations)
    for source_key_ptr in &sorted_node_keys {
      let source_key = *source_key_ptr; // source_key is &T
      if let Some(node) = self.nodes.get(source_key) {
        let mut sorted_successors: Vec<&T> = node.successors.iter().collect();
        sorted_successors.sort(); // T: Ord required for &T to sort

        for succ_key in sorted_successors {
          // succ_key is &T
          let mut is_immediate = true;
          let mut inner_sorted_successors_for_check: Vec<&T> = node.successors.iter().collect();
          inner_sorted_successors_for_check.sort(); // T: Ord required

          for intermediate_key in inner_sorted_successors_for_check {
            if intermediate_key == succ_key {
              continue;
            }
            if let Some(intermediate_node_w) = self.nodes.get(intermediate_key) {
              if intermediate_node_w.successors.contains(succ_key) {
                is_immediate = false;
                break;
              }
            }
          }

          if is_immediate {
            writeln!(
              file,
              "  \"{}\" -> \"{}\";",
              escape_dot_label(&source_key.to_string()), // T: Display
              escape_dot_label(&succ_key.to_string())    // T: Display
            )?;
          }
        }
      }
    }
    writeln!(file, "}}")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn m_lattice() -> Lattice<i32> {
    let mut m_lattice: Lattice<i32> = Lattice::new();
    // M-shape for non-unique meet if 5,6 considered uppers for join(1,3)
    //   5   6
    //  / \ / \
    // 1   2   3
    // join(1,3) -> should be None if 5 and 6 are incomparable & minimal uppers
    m_lattice.add_element(1);
    m_lattice.add_element(2);
    m_lattice.add_element(3);
    m_lattice.add_element(5);
    m_lattice.add_element(6);

    m_lattice.add_relation(1, 5);
    m_lattice.add_relation(2, 5);
    m_lattice.add_relation(2, 6);
    m_lattice.add_relation(3, 6);
    m_lattice
  }

  fn diamond_lattice() -> Lattice<i32> {
    // Create a diamond lattice:
    //     1
    //    / \
    //   2   3
    //    \ /
    //     4
    let mut diamond_lattice: Lattice<i32> = Lattice::new();
    diamond_lattice.add_element(1);
    diamond_lattice.add_element(2);
    diamond_lattice.add_element(3);
    diamond_lattice.add_element(4);

    diamond_lattice.add_relation(1, 2);
    diamond_lattice.add_relation(1, 3);
    diamond_lattice.add_relation(2, 4);
    diamond_lattice.add_relation(3, 4);

    diamond_lattice
  }

  #[test]
  fn test_basic_lattice() {
    let mut lattice = Lattice::new();
    lattice.add_relation(1, 2);
    lattice.add_relation(2, 3);

    assert!(lattice.leq(&1, &2).unwrap_or(false));
    assert!(lattice.leq(&2, &3).unwrap_or(false));
    assert!(lattice.leq(&1, &3).unwrap_or(false));
    assert!(!lattice.leq(&2, &1).unwrap_or(false));

    let minimal = lattice.minimal_elements();
    assert_eq!(minimal.len(), 1);
    assert!(minimal.contains(&1));

    let maximal = lattice.maximal_elements();
    assert_eq!(maximal.len(), 1);
    assert!(maximal.contains(&3));
  }

  #[test]
  fn test_diamond_lattice() {
    let lattice = diamond_lattice();
    assert!(lattice.leq(&1, &4).unwrap_or(false));
    assert!(!lattice.leq(&2, &3).unwrap_or(false));
    assert!(!lattice.leq(&3, &2).unwrap_or(false));

    let minimal = lattice.minimal_elements();
    assert_eq!(minimal.len(), 1);
    assert!(minimal.contains(&1));

    let maximal = lattice.maximal_elements();
    assert_eq!(maximal.len(), 1);
    assert!(maximal.contains(&4));
  }

  #[test]
  fn test_lattice_operations_diamond() {
    let mut lattice = Lattice::new();
    lattice.add_relation(1, 2);
    lattice.add_relation(1, 3);
    lattice.add_relation(2, 4);
    lattice.add_relation(3, 4);

    // Test join
    println!("join(2, 3): {:?}", lattice.join(2, 3));
    assert_eq!(lattice.join(2, 3), Some(4));

    println!("join(1, 4): {:?}", lattice.join(1, 4));
    assert_eq!(lattice.join(1, 4), Some(4));

    println!("join(1, 2): {:?}", lattice.join(1, 2));
    assert_eq!(lattice.join(1, 2), Some(2));

    println!("join(1, 1): {:?}", lattice.join(1, 1));
    assert_eq!(lattice.join(1, 1), Some(1));

    // Test meet
    println!("meet(2, 3): {:?}", lattice.meet(2, 3));
    assert_eq!(lattice.meet(2, 3), Some(1));

    println!("meet(1, 4): {:?}", lattice.meet(1, 4));
    assert_eq!(lattice.meet(1, 4), Some(1));

    println!("meet(2, 4): {:?}", lattice.meet(2, 4));
    assert_eq!(lattice.meet(2, 4), Some(2));

    println!("meet(4, 4): {:?}", lattice.meet(4, 4));
    assert_eq!(lattice.meet(4, 4), Some(4));
  }

  #[test]
  fn test_lattice_operations_non_lattice_examples() {
    let m_lattice = m_lattice();

    println!("join(1, 2) for M-shape: {:?}", m_lattice.join(1, 2));
    assert_eq!(m_lattice.join(1, 2), Some(5));

    println!("join(2, 3) for M-shape: {:?}", m_lattice.join(2, 3));
    assert_eq!(m_lattice.join(2, 3), Some(6));

    println!("join(1, 3) for M-shape: {:?}", m_lattice.join(1, 3));
    assert_eq!(m_lattice.join(1, 3), None);

    // Example with two minimal upper bounds:
    //   c   d
    //  / \ /
    // a   b
    let mut non_join_lattice: Lattice<&str> = Lattice::new();
    non_join_lattice.add_relation("a", "c");
    non_join_lattice.add_relation("a", "d");
    non_join_lattice.add_relation("b", "c");
    non_join_lattice.add_relation("b", "d");

    assert_eq!(non_join_lattice.join("a", "b"), None);
    assert_eq!(non_join_lattice.meet("c", "d"), None);

    // Example with two maximal lower bounds:
    //   a   b
    //  / \ / \
    // c   d
    let mut non_meet_lattice: Lattice<&str> = Lattice::new();
    non_meet_lattice.add_relation("c", "a");
    non_meet_lattice.add_relation("d", "a");
    non_meet_lattice.add_relation("c", "b");
    non_meet_lattice.add_relation("d", "b");

    assert_eq!(non_meet_lattice.meet("a", "b"), None);
    assert_eq!(non_meet_lattice.join("c", "d"), None);
  }

  #[test]
  fn test_graphviz_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let temp_path = temp_dir.path().join("test_m_shape_lattice.dot");
    let filename = temp_path.to_str().unwrap();
    println!("--- M-shape Example - Saving to {filename} ---");
    m_lattice().save_to_dot_file(filename).expect("Failed to save M-shape lattice");

    // Check if the file was created
    assert!(temp_path.exists());

    // Clean up the temporary directory
    drop(temp_dir);
  }

  #[test]
  #[ignore = "Manual test to see output of M-shape lattice"]
  fn test_graphviz_output_manual() {
    let filename = "test_m_shape_lattice.dot";
    println!("--- M-shape Example - Saving to {filename} ---");
    m_lattice().save_to_dot_file(filename).expect("Failed to save M-shape lattice");
  }

  #[test]
  fn test_downset() {
    let lattice = diamond_lattice();
    let downset = lattice.downset(4);
    assert_eq!(downset, HashSet::from([1, 2, 3, 4]));

    let downset = lattice.downset(2);
    assert_eq!(downset, HashSet::from([1, 2]));

    let downset = lattice.downset(1);
    assert_eq!(downset, HashSet::from([1]));
  }

  #[test]
  fn test_upset() {
    let lattice = diamond_lattice();
    let upset = lattice.upset(4);
    assert_eq!(upset, HashSet::from([4]));

    let upset = lattice.upset(2);
    assert_eq!(upset, HashSet::from([2, 4]));

    let upset = lattice.upset(1);
    assert_eq!(upset, HashSet::from([1, 2, 3, 4]));
  }

  #[test]
  fn test_successors() {
    let lattice = diamond_lattice();
    let successors = lattice.successors(1);
    assert_eq!(successors, HashSet::from([2, 3]));
  }

  #[test]
  fn test_predecessors() {
    let lattice = diamond_lattice();
    let predecessors = lattice.predecessors(4);
    assert_eq!(predecessors, HashSet::from([2, 3]));
  }
}
