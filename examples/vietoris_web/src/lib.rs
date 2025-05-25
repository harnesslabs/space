//! # Interactive Vietoris-Rips Complex Demo
//!
//! This is a WASM-based web demo that allows interactive exploration of Vietoris-Rips complexes.
//! Users can:
//! - Click to add points to a 2D plane
//! - Right-click to remove points
//! - Adjust epsilon with a slider to see how the complex changes
//! - Visualize vertices (0-simplices), edges (1-simplices), and triangles (2-simplices)

#[cfg(feature = "wasm")] use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys::{console, CanvasRenderingContext2d};

// When the `console_error_panic_hook` feature is enabled, we can call the
// `set_panic_hook` function at least once during initialization, and then
// we will get better error messages if our code ever panics.
#[cfg(all(feature = "wasm", feature = "console_error_panic_hook"))]
extern crate console_error_panic_hook;

/// Represents a point in the 2D plane
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct Point2D {
  pub x: f64,
  pub y: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl Point2D {
  #[wasm_bindgen(constructor)]
  pub fn new(x: f64, y: f64) -> Point2D { Point2D { x, y } }

  #[wasm_bindgen(getter)]
  pub fn x(&self) -> f64 { self.x }

  #[wasm_bindgen(getter)]
  pub fn y(&self) -> f64 { self.y }
}

/// Simple edge representation
#[cfg(feature = "wasm")]
#[derive(Clone, Debug)]
pub struct Edge {
  pub start: usize,
  pub end:   usize,
}

/// Simple triangle representation
#[cfg(feature = "wasm")]
#[derive(Clone, Debug)]
pub struct Triangle {
  pub a: usize,
  pub b: usize,
  pub c: usize,
}

/// Main demo structure that manages the interactive Vietoris-Rips visualization
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct VietorisRipsDemo {
  points:        Vec<Point2D>,
  epsilon:       f64,
  canvas_width:  f64,
  canvas_height: f64,
}

#[cfg(feature = "wasm")]
impl VietorisRipsDemo {
  /// Calculate distance between two points
  fn distance(&self, i: usize, j: usize) -> f64 {
    let p1 = &self.points[i];
    let p2 = &self.points[j];
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
  }

  /// Build edges for the current epsilon
  fn build_edges(&self) -> Vec<Edge> {
    let mut edges = Vec::new();
    for i in 0..self.points.len() {
      for j in (i + 1)..self.points.len() {
        if self.distance(i, j) <= self.epsilon {
          edges.push(Edge { start: i, end: j });
        }
      }
    }
    edges
  }

  /// Build triangles for the current epsilon
  fn build_triangles(&self) -> Vec<Triangle> {
    let mut triangles = Vec::new();
    for i in 0..self.points.len() {
      for j in (i + 1)..self.points.len() {
        for k in (j + 1)..self.points.len() {
          // Check if all three edges exist
          if self.distance(i, j) <= self.epsilon
            && self.distance(j, k) <= self.epsilon
            && self.distance(i, k) <= self.epsilon
          {
            triangles.push(Triangle { a: i, b: j, c: k });
          }
        }
      }
    }
    triangles
  }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl VietorisRipsDemo {
  /// Create a new demo instance
  #[wasm_bindgen(constructor)]
  pub fn new(canvas_width: f64, canvas_height: f64) -> Self {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    Self {
      points: Vec::new(),
      epsilon: 50.0, // Default epsilon
      canvas_width,
      canvas_height,
    }
  }

  /// Add a point at the given coordinates
  #[wasm_bindgen]
  pub fn add_point(&mut self, x: f64, y: f64) {
    let point = Point2D::new(x, y);
    self.points.push(point);
    console::log_1(
      &format!("Added point at ({}, {}). Total points: {}", x, y, self.points.len()).into(),
    );
  }

  /// Remove the point closest to the given coordinates (within a threshold)
  #[wasm_bindgen]
  pub fn remove_point(&mut self, x: f64, y: f64) -> bool {
    const REMOVE_THRESHOLD: f64 = 20.0; // pixels

    if let Some((index, _)) = self
      .points
      .iter()
      .enumerate()
      .map(|(i, p)| (i, (p.x - x).hypot(p.y - y)))
      .filter(|(_, dist)| *dist < REMOVE_THRESHOLD)
      .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
    {
      self.points.remove(index);
      console::log_1(&format!("Removed point. Total points: {}", self.points.len()).into());
      true
    } else {
      false
    }
  }

  /// Set the epsilon value for the Vietoris-Rips complex
  #[wasm_bindgen]
  pub fn set_epsilon(&mut self, epsilon: f64) {
    self.epsilon = epsilon;
    console::log_1(&format!("Set epsilon to: {}", epsilon).into());
  }

  /// Get the current epsilon value
  #[wasm_bindgen]
  pub fn get_epsilon(&self) -> f64 { self.epsilon }

  /// Get the number of points
  #[wasm_bindgen]
  pub fn point_count(&self) -> usize { self.points.len() }

  /// Clear all points
  #[wasm_bindgen]
  pub fn clear_points(&mut self) {
    self.points.clear();
    console::log_1(&"Cleared all points".into());
  }

  /// Build the Vietoris-Rips complex and return statistics
  #[wasm_bindgen]
  pub fn get_complex_stats(&self) -> ComplexStats {
    if self.points.is_empty() {
      return ComplexStats::new(0, 0, 0);
    }

    let edges = self.build_edges();
    let triangles = self.build_triangles();

    ComplexStats::new(self.points.len(), edges.len(), triangles.len())
  }

  /// Render the current state to the canvas
  #[wasm_bindgen]
  pub fn render(&self, context: &CanvasRenderingContext2d) {
    // Clear canvas
    context.clear_rect(0.0, 0.0, self.canvas_width, self.canvas_height);

    if self.points.is_empty() {
      return;
    }

    // Build the complex
    let edges = self.build_edges();
    let triangles = self.build_triangles();

    // Render triangles (2-simplices) first (so they appear behind edges and vertices)
    self.render_triangles(&context, &triangles);

    // Render edges (1-simplices)
    self.render_edges(&context, &edges);

    // Render vertices (0-simplices) last (so they appear on top)
    self.render_vertices(&context);
  }

  /// Render vertices as circles
  fn render_vertices(&self, context: &CanvasRenderingContext2d) {
    context.set_fill_style(&"#2563eb".into()); // Blue
    context.set_stroke_style(&"#1e40af".into()); // Darker blue
    context.set_line_width(2.0);

    for point in &self.points {
      context.begin_path();
      context.arc(point.x, point.y, 6.0, 0.0, 2.0 * std::f64::consts::PI).unwrap();
      context.fill();
      context.stroke();
    }
  }

  /// Render edges as lines
  fn render_edges(&self, context: &CanvasRenderingContext2d, edges: &[Edge]) {
    context.set_stroke_style(&"#059669".into()); // Green
    context.set_line_width(2.0);

    for edge in edges {
      let p1 = &self.points[edge.start];
      let p2 = &self.points[edge.end];

      context.begin_path();
      context.move_to(p1.x, p1.y);
      context.line_to(p2.x, p2.y);
      context.stroke();
    }
  }

  /// Render triangles as filled shapes
  fn render_triangles(&self, context: &CanvasRenderingContext2d, triangles: &[Triangle]) {
    context.set_fill_style(&"rgba(239, 68, 68, 0.3)".into()); // Semi-transparent red
    context.set_stroke_style(&"#dc2626".into()); // Red border
    context.set_line_width(1.0);

    for triangle in triangles {
      let p1 = &self.points[triangle.a];
      let p2 = &self.points[triangle.b];
      let p3 = &self.points[triangle.c];

      context.begin_path();
      context.move_to(p1.x, p1.y);
      context.line_to(p2.x, p2.y);
      context.line_to(p3.x, p3.y);
      context.close_path();
      context.fill();
      context.stroke();
    }
  }
}

/// Statistics about the current complex
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct ComplexStats {
  vertices:  usize,
  edges:     usize,
  triangles: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ComplexStats {
  #[wasm_bindgen(constructor)]
  pub fn new(vertices: usize, edges: usize, triangles: usize) -> ComplexStats {
    ComplexStats { vertices, edges, triangles }
  }

  #[wasm_bindgen(getter)]
  pub fn vertices(&self) -> usize { self.vertices }

  #[wasm_bindgen(getter)]
  pub fn edges(&self) -> usize { self.edges }

  #[wasm_bindgen(getter)]
  pub fn triangles(&self) -> usize { self.triangles }
}

/// Initialize the demo - call this from JavaScript
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn main() {
  #[cfg(feature = "console_error_panic_hook")]
  console_error_panic_hook::set_once();

  console::log_1(&"Vietoris-Rips Demo initialized!".into());
}
