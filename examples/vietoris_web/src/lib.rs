//! # Interactive Vietoris-Rips Complex Demo
//!
//! A WebAssembly library that provides an interactive demonstration of Vietoris-Rips complexes
//! using the `cova` library.
//!
//! ## Features
//! - Interactive point placement and removal
//! - Real-time Vietoris-Rips complex computation
//! - Canvas-based visualization of simplicial complexes
//! - Adjustable epsilon parameter

#![cfg(target_arch = "wasm32")]

use cova::{
  algebra::tensors::fixed::FixedVector,
  space::{
    cloud::Cloud,
    complexes::SimplicialComplex,
    filtration::{vietoris_rips::VietorisRips, Filtration},
  },
};
use wasm_bindgen::prelude::*;
use web_sys::{console, CanvasRenderingContext2d};

// Enable better error messages in debug mode
extern crate console_error_panic_hook;

/// A point in 2D space
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct Point2D {
  x: f64,
  y: f64,
}

#[wasm_bindgen]
impl Point2D {
  #[wasm_bindgen(constructor)]
  pub fn new(x: f64, y: f64) -> Point2D { Point2D { x, y } }

  #[wasm_bindgen(getter)]
  pub fn x(&self) -> f64 { self.x }

  #[wasm_bindgen(getter)]
  pub fn y(&self) -> f64 { self.y }
}

/// Statistics about the current simplicial complex
#[wasm_bindgen]
pub struct ComplexStats {
  vertices:  usize,
  edges:     usize,
  triangles: usize,
}
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

/// Main demo structure that manages the interactive Vietoris-Rips visualization
#[wasm_bindgen]
pub struct VietorisRipsDemo {
  points:        Vec<Point2D>,
  epsilon:       f64,
  canvas_width:  f64,
  canvas_height: f64,
  vr_builder:    VietorisRips<2, f64, SimplicialComplex>,
}
#[wasm_bindgen]
impl VietorisRipsDemo {
  /// Create a new demo instance
  #[wasm_bindgen(constructor)]
  pub fn new(canvas_width: f64, canvas_height: f64) -> Self {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    Self {
      points: Vec::new(),
      epsilon: 50.0,
      canvas_width,
      canvas_height,
      vr_builder: VietorisRips::new(),
    }
  }

  /// Add a point at the given coordinates
  #[wasm_bindgen]
  pub fn add_point(&mut self, x: f64, y: f64) {
    let point = Point2D::new(x, y);
    self.points.push(point);
    console::log_1(&format!("Added point at ({}, {}). Total: {}", x, y, self.points.len()).into());
  }

  /// Remove the point closest to the given coordinates (within threshold)
  #[wasm_bindgen]
  pub fn remove_point(&mut self, x: f64, y: f64) -> bool {
    const REMOVE_THRESHOLD: f64 = 20.0;

    if let Some((index, _)) = self
      .points
      .iter()
      .enumerate()
      .map(|(i, p)| (i, (p.x - x).hypot(p.y - y)))
      .filter(|(_, dist)| *dist < REMOVE_THRESHOLD)
      .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
    {
      self.points.remove(index);
      console::log_1(&format!("Removed point. Total: {}", self.points.len()).into());
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

  /// Convert points to FixedVector format for cova
  fn points_to_vectors(&self) -> Vec<FixedVector<2, f64>> {
    self.points.iter().map(|p| FixedVector([p.x, p.y])).collect()
  }

  /// Build the Vietoris-Rips complex and return statistics
  #[wasm_bindgen]
  pub fn get_complex_stats(&self) -> ComplexStats {
    if self.points.is_empty() {
      return ComplexStats::new(0, 0, 0);
    }

    let cloud_points = self.points_to_vectors();
    let cloud = Cloud::<2, f64>::new(cloud_points);
    let complex = self.vr_builder.build(&cloud, self.epsilon, &());

    let vertices = complex.elements_of_dimension(0).len();
    let edges = complex.elements_of_dimension(1).len();
    let triangles = complex.elements_of_dimension(2).len();

    ComplexStats::new(vertices, edges, triangles)
  }

  /// Render the current state to the canvas
  #[wasm_bindgen]
  pub fn render(&self, context: &CanvasRenderingContext2d) {
    // Clear canvas
    context.clear_rect(0.0, 0.0, self.canvas_width, self.canvas_height);

    if self.points.is_empty() {
      return;
    }

    // Build the complex using cova
    let cloud_points = self.points_to_vectors();
    let cloud = Cloud::<2, f64>::new(cloud_points);
    let complex = self.vr_builder.build(&cloud, self.epsilon, &());

    // Render in order: triangles, edges, vertices (back to front)
    self.render_triangles(&context, &complex);
    self.render_edges(&context, &complex);
    self.render_vertices(&context);
  }

  /// Render vertices as blue circles
  fn render_vertices(&self, context: &CanvasRenderingContext2d) {
    context.set_fill_style(&"#2563eb".into());
    context.set_stroke_style(&"#1e40af".into());
    context.set_line_width(2.0);

    for point in &self.points {
      context.begin_path();
      context.arc(point.x, point.y, 6.0, 0.0, 2.0 * std::f64::consts::PI).unwrap();
      context.fill();
      context.stroke();
    }
  }

  /// Render edges as green lines
  fn render_edges(&self, context: &CanvasRenderingContext2d, complex: &SimplicialComplex) {
    context.set_stroke_style(&"#059669".into());
    context.set_line_width(2.0);

    let edges = complex.elements_of_dimension(1);
    for edge in edges {
      let vertices = edge.vertices();
      if vertices.len() == 2 {
        let p1 = &self.points[vertices[0]];
        let p2 = &self.points[vertices[1]];

        context.begin_path();
        context.move_to(p1.x, p1.y);
        context.line_to(p2.x, p2.y);
        context.stroke();
      }
    }
  }

  /// Render triangles as semi-transparent red shapes
  fn render_triangles(&self, context: &CanvasRenderingContext2d, complex: &SimplicialComplex) {
    context.set_fill_style(&"rgba(239, 68, 68, 0.3)".into());
    context.set_stroke_style(&"#dc2626".into());
    context.set_line_width(1.0);

    let triangles = complex.elements_of_dimension(2);
    for triangle in triangles {
      let vertices = triangle.vertices();
      if vertices.len() == 3 {
        let p1 = &self.points[vertices[0]];
        let p2 = &self.points[vertices[1]];
        let p3 = &self.points[vertices[2]];

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
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
  #[cfg(feature = "console_error_panic_hook")]
  console_error_panic_hook::set_once();

  console::log_1(&"🦀 Vietoris-Rips Demo WASM module initialized!".into());
}
