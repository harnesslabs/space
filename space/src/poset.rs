use std::{
  collections::{HashMap, HashSet},
  fmt::{Display, Formatter},
  hash::Hash,
};

use termgraph::{Config, DirectedGraph, ValueFormatter};

/// A node in a poset representing an element and its relationships
#[derive(Debug, Clone)]
pub struct PosetNode<T> {
  /// The element stored in this node
  element:      T,
  /// Direct successors (elements that are greater than this one)
  successors:   HashSet<T>,
  /// Direct predecessors (elements that are less than this one)
  predecessors: HashSet<T>,
}

/// A general poset structure that can represent any partially ordered set
#[derive(Debug)]
pub struct GeneralPoset<T> {
  /// Map of elements to their nodes
  nodes: HashMap<T, PosetNode<T>>,
}

impl<T: Hash + Eq + Clone> GeneralPoset<T> {
  /// Creates a new empty poset
  pub fn new() -> Self { Self { nodes: HashMap::new() } }

  /// Adds a new element to the poset
  pub fn add_element(&mut self, element: T) {
    if !self.nodes.contains_key(&element) {
      self.nodes.insert(element.clone(), PosetNode {
        element,
        successors: HashSet::new(),
        predecessors: HashSet::new(),
      });
    }
  }

  /// Adds a relation a ≤ b to the poset
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

  /// Computes the transitive closure of the poset
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

  /// Checks if a ≤ b in the poset
  pub fn leq(&self, a: &T, b: &T) -> bool {
    if let Some(node_a) = self.nodes.get(a) {
      node_a.successors.contains(b)
    } else {
      false
    }
  }

  /// Returns all minimal elements in the poset
  pub fn minimal_elements(&self) -> HashSet<T> {
    self
      .nodes
      .iter()
      .filter(|(_, node)| node.predecessors.is_empty())
      .map(|(element, _)| element.clone())
      .collect()
  }

  /// Returns all maximal elements in the poset
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
      return None; // Elements must be in the poset
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
      return None; // Elements must be in the poset
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

impl<T: Hash + Eq + Clone + Display> std::fmt::Display for GeneralPoset<T> {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    if self.nodes.is_empty() {
      return writeln!(f, "Empty Poset");
    }

    let mut node_to_id = HashMap::new();
    let mut next_id: usize = 0;
    let mut termgraph_nodes_with_id = Vec::new();

    // Assign usize IDs to each node T and prepare (ID, Label) tuples for termgraph
    for t_node_val in self.nodes.keys() {
      let id = *node_to_id.entry(t_node_val.clone()).or_insert_with(|| {
        let current_id = next_id;
        next_id += 1;
        current_id
      });
      termgraph_nodes_with_id.push((id, t_node_val.to_string()));
    }

    let mut termgraph_edges_with_id = Vec::new();
    // Find covering relations (Hasse diagram edges)
    for (source_element, node) in &self.nodes {
      if let Some(source_id) = node_to_id.get(source_element) {
        for succ in &node.successors {
          // Check if 'succ' is an immediate successor of 'source_element'
          let mut is_immediate = true;
          for intermediate_w in &node.successors {
            // intermediate_w is > source_element
            if intermediate_w == succ {
              continue; // Don't check against self
            }
            // Check if source_element < intermediate_w < succ
            if let Some(intermediate_node_w) = self.nodes.get(intermediate_w) {
              if intermediate_node_w.successors.contains(succ) {
                is_immediate = false;
                break;
              }
            }
          }

          if is_immediate {
            if let Some(target_id) = node_to_id.get(succ) {
              termgraph_edges_with_id.push((*source_id, *target_id));
            }
          }
        }
      }
    }

    let mut graph = DirectedGraph::new();
    graph.add_nodes(termgraph_nodes_with_id);
    graph.add_edges(termgraph_edges_with_id);

    // Configure termgraph (similar to the example)
    // Using a max_width of 5, adjust as needed.
    let config = Config::new(ValueFormatter::new(), 5);
    let mut buffer = Vec::new();

    // termgraph::fdisplay is assumed to return () and handle its own IO errors (e.g., by panic).
    termgraph::fdisplay(&graph, &config, &mut buffer);

    // Convert the buffer to a String and write it to the Formatter 'f'.
    // This is where our fmt::Result is determined.
    String::from_utf8(buffer)
        .map_err(|_| std::fmt::Error) // Convert UTF8 error to fmt::Error
        .and_then(|s| f.write_str(&s)) // Write string to formatter, f.write_str returns fmt::Result
  }
}

#[cfg(test)]
mod tests {
  use std::io::Cursor;

  use super::*;

  #[test]
  fn test_basic_poset() {
    let mut poset = GeneralPoset::new();

    // Create a simple poset: 1 ≤ 2 ≤ 3
    poset.add_relation(1, 2);
    poset.add_relation(2, 3);

    println!("{poset}");

    assert!(poset.leq(&1, &2));
    assert!(poset.leq(&2, &3));
    assert!(poset.leq(&1, &3)); // Transitive closure
    assert!(!poset.leq(&2, &1));

    let minimal = poset.minimal_elements();
    assert_eq!(minimal.len(), 1);
    assert!(minimal.contains(&1));

    let maximal = poset.maximal_elements();
    assert_eq!(maximal.len(), 1);
    assert!(maximal.contains(&3));
  }

  #[test]
  fn test_diamond_poset() {
    let mut poset = GeneralPoset::new();

    // Create a diamond poset:
    //     1
    //    / \
    //   2   3
    //    \ /
    //     4
    poset.add_relation(1, 2);
    poset.add_relation(1, 3);
    poset.add_relation(2, 4);
    poset.add_relation(3, 4);

    println!("--- Diamond Poset for Basic Tests ---");
    println!("{}", poset);

    assert!(poset.leq(&1, &4));
    assert!(!poset.leq(&2, &3));
    assert!(!poset.leq(&3, &2));

    let minimal = poset.minimal_elements();
    assert_eq!(minimal.len(), 1);
    assert!(minimal.contains(&1));

    let maximal = poset.maximal_elements();
    assert_eq!(maximal.len(), 1);
    assert!(maximal.contains(&4));
  }

  #[test]
  fn test_lattice_operations_diamond() {
    let mut poset = GeneralPoset::new();
    poset.add_relation(1, 2); // Element type is inferred as i32
    poset.add_relation(1, 3);
    poset.add_relation(2, 4);
    poset.add_relation(3, 4);

    println!("--- Diamond Poset for Lattice Operations ---");
    println!("{poset}");

    // Test join
    println!("join(2, 3): {:?}", poset.join(2, 3));
    assert_eq!(poset.join(2, 3), Some(4));

    println!("join(1, 4): {:?}", poset.join(1, 4));
    assert_eq!(poset.join(1, 4), Some(4)); // 1 <= 4, 4 <= 4 -> LUB is 4

    println!("join(1, 2): {:?}", poset.join(1, 2));
    assert_eq!(poset.join(1, 2), Some(2)); // 1 <= 2, 2 <= 2 -> LUB is 2

    println!("join(1, 1): {:?}", poset.join(1, 1));
    assert_eq!(poset.join(1, 1), Some(1));

    // Test meet
    println!("meet(2, 3): {:?}", poset.meet(2, 3));
    assert_eq!(poset.meet(2, 3), Some(1));

    println!("meet(1, 4): {:?}", poset.meet(1, 4));
    assert_eq!(poset.meet(1, 4), Some(1)); // 1 <= 1, 1 <= 4 -> GLB is 1

    println!("meet(2, 4): {:?}", poset.meet(2, 4));
    assert_eq!(poset.meet(2, 4), Some(2)); // 2 <= 2, 2 <= 4 -> GLB is 2

    println!("meet(4, 4): {:?}", poset.meet(4, 4));
    assert_eq!(poset.meet(4, 4), Some(4));

    // Test elements not in poset (though add_relation adds them)
    // To properly test this part of join/meet, we'd need to add elements without relations first.
    // For now, this primarily tests the logic assuming elements are present.
    // println!("join(5, 6): {:?}", poset.join(5, 6));
    // assert_eq!(poset.join(5, 6), None);
  }

  #[test]
  fn test_lattice_operations_non_lattice() {
    let mut poset: GeneralPoset<i32> = GeneralPoset::new();
    //   1   2  (incomparable)
    //   |   |
    //   3   4  (incomparable)
    // No common upper bounds for 3,4 other than 1 and 2 if we add 3<1, 4<2
    // No common lower bounds for 1,2 other than 3 and 4

    // Create a poset where join/meet might not be unique
    // M-shape for non-unique meet:
    //   5   6
    //  / \ / \
    // 1   2   3
    // join(1,3) -> should be None if 5 and 6 are incomparable
    poset.add_element(1);
    poset.add_element(2);
    poset.add_element(3);
    poset.add_element(5);
    poset.add_element(6);

    poset.add_relation(1, 5);
    poset.add_relation(2, 5);
    poset.add_relation(2, 6);
    poset.add_relation(3, 6);

    println!("--- M-shape Poset for Non-Lattice Operations ---");
    println!("{}", poset);

    println!("join(1, 2): {:?}", poset.join(1, 2)); // Should be Some(5)
    assert_eq!(poset.join(1, 2), Some(5));

    println!("join(2, 3): {:?}", poset.join(2, 3)); // Should be Some(6)
    assert_eq!(poset.join(2, 3), Some(6));

    println!("join(1, 3): {:?}", poset.join(1, 3)); // Upper bounds: {5,6}. If 5,6 incomparable, minimal are {5,6}. So None.
    assert_eq!(poset.join(1, 3), None);

    // N-shape for non-unique join:
    // 1   2
    // \ / \
    //   3   4
    //  / \ /
    // 5   6
    // meet(1,2) -> should be None if 3 and 4 are incomparable
    let mut poset2: GeneralPoset<i32> = GeneralPoset::new();
    poset2.add_element(1);
    poset2.add_element(2);
    poset2.add_element(3);
    poset2.add_element(4);
    poset2.add_element(5);
    poset2.add_element(6);

    poset2.add_relation(3, 1);
    poset2.add_relation(4, 1);
    poset2.add_relation(4, 2);
    poset2.add_relation(5, 2); // Mistake: should be 5 -> 2 for N shape. Correcting to 5->4 or similar or 3,5 -> 1; 4,6 -> 2

    // Let's simplify the non-lattice test for clarity.
    // Two elements with two minimal upper bounds:
    //   c   d
    //  / \ /
    // a   b
    let mut poset_non_join: GeneralPoset<&str> = GeneralPoset::new();
    poset_non_join.add_relation("a", "c");
    poset_non_join.add_relation("a", "d");
    poset_non_join.add_relation("b", "c");
    poset_non_join.add_relation("b", "d");
    // Assume "c" and "d" are incomparable. compute_transitive_closure won't make them related.

    println!("--- Non-Unique Join Poset ---");
    println!("{}", poset_non_join);
    println!("join(\"a\", \"b\"): {:?}", poset_non_join.join("a", "b")); // Common uppers: {c, d}. Minimals: {c,d} -> None
    assert_eq!(poset_non_join.join("a", "b"), None);
    println!("meet(\"c\", \"d\"): {:?}", poset_non_join.meet("c", "d")); // Common lowers: {a, b}. Maximals: {a,b} -> None
    assert_eq!(poset_non_join.meet("c", "d"), None);

    // Two elements with two maximal lower bounds:
    //   a   b
    //  / \ / \
    // c   d
    let mut poset_non_meet: GeneralPoset<&str> = GeneralPoset::new();
    poset_non_meet.add_relation("c", "a");
    poset_non_meet.add_relation("d", "a");
    poset_non_meet.add_relation("c", "b");
    poset_non_meet.add_relation("d", "b");
    // Assume "c" and "d" are incomparable.

    println!("--- Non-Unique Meet Poset ---");
    println!("{}", poset_non_meet);
    println!("meet(\"a\", \"b\"): {:?}", poset_non_meet.meet("a", "b")); // Common lowers: {c, d}. Maximals: {c,d} -> None
    assert_eq!(poset_non_meet.meet("a", "b"), None);
    println!("join(\"c\", \"d\"): {:?}", poset_non_meet.join("c", "d")); // Common uppers: {a, b}. Minimals: {a,b} -> None
    assert_eq!(poset_non_meet.join("c", "d"), None);
  }
}
