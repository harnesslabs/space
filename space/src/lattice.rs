use std::{
  collections::{HashMap, HashSet},
  fmt::Write as FmtWrite,
  fs::File,
  hash::Hash,
  io::{Result as IoResult, Write as IoWrite},
};

/// A node in a lattice representing an element and its relationships
#[derive(Debug, Clone)]
pub struct LatticeNode<T> {
  /// The element stored in this node
  element:      T,
  /// Direct successors (elements that are greater than this one)
  successors:   HashSet<T>,
  /// Direct predecessors (elements that are less than this one)
  predecessors: HashSet<T>,
}

/// A general lattice structure that can represent any partially ordered set
/// with join and meet operations.
#[derive(Debug)]
pub struct Lattice<T> {
  /// Map of elements to their nodes
  nodes: HashMap<T, LatticeNode<T>>,
}

impl<T: Hash + Eq + Clone> Lattice<T> {
  /// Creates a new empty lattice
  pub fn new() -> Self { Self { nodes: HashMap::new() } }

  /// Adds a new element to the lattice
  pub fn add_element(&mut self, element: T) {
    if !self.nodes.contains_key(&element) {
      self.nodes.insert(element.clone(), LatticeNode {
        element,
        successors: HashSet::new(),
        predecessors: HashSet::new(),
      });
    }
  }

  /// Adds a relation a ≤ b to the lattice
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

  /// Checks if a ≤ b in the lattice
  pub fn leq(&self, a: &T, b: &T) -> bool {
    if let Some(node_a) = self.nodes.get(a) {
      node_a.successors.contains(b)
    } else {
      false
    }
  }

  /// Returns all minimal elements in the lattice
  pub fn minimal_elements(&self) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| node.predecessors.is_empty())
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Returns all maximal elements in the lattice
  pub fn maximal_elements(&self) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| node.successors.is_empty())
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Computes the join (least upper bound) of two elements a and b.
  /// Returns None if the join does not exist or is not unique.
  pub fn join(&self, a: T, b: T) -> Option<T> {
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
      .filter(|&x| common_upper_bounds.iter().all(|y| x == y || !self.leq(y, x)))
      .cloned()
      .collect();

    if minimal_common_upper_bounds.len() == 1 {
      Some(minimal_common_upper_bounds[0].clone())
    } else {
      None
    }
  }

  /// Computes the meet (greatest lower bound) of two elements a and b.
  /// Returns None if the meet does not exist or is not unique.
  pub fn meet(&self, a: T, b: T) -> Option<T> {
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
      .filter(|&x| common_lower_bounds.iter().all(|y| x == y || !self.leq(x, y)))
      .cloned()
      .collect();

    if maximal_common_lower_bounds.len() == 1 {
      Some(maximal_common_lower_bounds[0].clone())
    } else {
      None
    }
  }
}

// Helper function to escape strings for DOT format
fn escape_dot_label(label: &str) -> String { label.replace('"', "\\\"") }

// Implementation block for methods requiring Display and Ord for T
impl<T: Hash + Eq + Clone + std::fmt::Display + Ord> Lattice<T> {
  /// Saves the lattice representation in DOT format to the specified file.
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

  #[test]
  fn test_basic_lattice() {
    let mut lattice = Lattice::new();
    lattice.add_relation(1, 2);
    lattice.add_relation(2, 3);

    assert!(lattice.leq(&1, &2));
    assert!(lattice.leq(&2, &3));
    assert!(lattice.leq(&1, &3));
    assert!(!lattice.leq(&2, &1));

    let minimal = lattice.minimal_elements();
    assert_eq!(minimal.len(), 1);
    assert!(minimal.contains(&1));

    let maximal = lattice.maximal_elements();
    assert_eq!(maximal.len(), 1);
    assert!(maximal.contains(&3));
  }

  #[test]
  fn test_diamond_lattice() {
    let mut lattice = Lattice::new();

    // Create a diamond lattice:
    //     1
    //    / \
    //   2   3
    //    \ /
    //     4
    lattice.add_relation(1, 2);
    lattice.add_relation(1, 3);
    lattice.add_relation(2, 4);
    lattice.add_relation(3, 4);

    assert!(lattice.leq(&1, &4));
    assert!(!lattice.leq(&2, &3));
    assert!(!lattice.leq(&3, &2));

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
}
