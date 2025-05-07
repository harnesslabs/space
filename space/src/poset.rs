use std::{
  collections::{HashMap, HashSet},
  fmt::{Display, Formatter},
  hash::Hash,
  io::Write,
};

use itertools::Itertools;
use termgraph::{Config, DirectedGraph, IDFormatter, ValueFormatter};

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

    println!("{poset}");

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
}
