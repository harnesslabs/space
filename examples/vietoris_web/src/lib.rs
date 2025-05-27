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
  prelude::*,
  space::{
    cloud::Cloud,
    complexes::SimplicialComplex,
    filtration::{Filtration, vietoris_rips::VietorisRips},
  },
};
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, console};

// Enable better error messages in debug mode
extern crate console_error_panic_hook;

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
  cloud:         Cloud<2>,
  epsilon:       f64,
  canvas_width:  f64,
  canvas_height: f64,
  vr_builder:    VietorisRips<2, SimplicialComplex>,
}

#[wasm_bindgen]
impl VietorisRipsDemo {
  /// Create a new demo instance
  #[wasm_bindgen(constructor)]
  pub fn new(canvas_width: f64, canvas_height: f64) -> Self {
    console_error_panic_hook::set_once();

    Self {
      cloud: Cloud::new(Vec::new()),
      epsilon: 50.0,
      canvas_width,
      canvas_height,
      vr_builder: VietorisRips::new(),
    }
  }

  /// Add a point at the given coordinates
  #[wasm_bindgen]
  pub fn add_point(&mut self, x: f64, y: f64) {
    let point = FixedVector::<2, f64>::from([x, y]);
    let mut points = self.cloud.points_ref().to_vec();
    points.push(point);
    self.cloud = Cloud::new(points);

    console::log_1(
      &format!("Added point at ({}, {}). Total: {}", x, y, self.cloud.points_ref().len()).into(),
    );
  }

  /// Remove the point closest to the given coordinates (within threshold)
  #[wasm_bindgen]
  pub fn remove_point(&mut self, x: f64, y: f64) -> bool {
    const REMOVE_THRESHOLD: f64 = 20.0;

    let points = self.cloud.points_ref();
    if let Some((index, _)) = points
      .iter()
      .enumerate()
      .map(|(i, p)| (i, (p.0[0] - x).hypot(p.0[1] - y)))
      .filter(|(_, dist)| *dist < REMOVE_THRESHOLD)
      .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
    {
      let mut new_points = points.to_vec();
      new_points.remove(index);
      self.cloud = Cloud::new(new_points);

      console::log_1(&format!("Removed point. Total: {}", self.cloud.points_ref().len()).into());
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
  pub fn point_count(&self) -> usize { self.cloud.points_ref().len() }

  /// Clear all points
  #[wasm_bindgen]
  pub fn clear_points(&mut self) {
    self.cloud = Cloud::new(Vec::new());
    console::log_1(&"Cleared all points".into());
  }

  /// Build the Vietoris-Rips complex and return statistics
  #[wasm_bindgen]
  pub fn get_complex_stats(&self) -> ComplexStats {
    if self.cloud.is_empty() {
      return ComplexStats::new(0, 0, 0);
    }

    let complex = self.vr_builder.build(&self.cloud, self.epsilon, &());

    let vertices = complex.elements_of_dimension(0).len();
    let edges = complex.elements_of_dimension(1).len();
    let triangles = complex.elements_of_dimension(2).len();

    ComplexStats::new(vertices, edges, triangles)
  }

  /// Render the current state to the canvas
  #[wasm_bindgen]
  pub fn render(&self, context: &CanvasRenderingContext2d) {
    // Clear canvas with white background
    context.set_fill_style(&JsValue::from_str("#ffffff"));
    context.fill_rect(0.0, 0.0, self.canvas_width, self.canvas_height);

    if self.cloud.is_empty() {
      return;
    }

    // Build the complex using cova
    let complex = self.vr_builder.build(&self.cloud, self.epsilon, &());

    // Render in order: epsilon bubbles, triangles, edges, vertices (back to front)
    self.render_epsilon_bubbles(&context);
    self.render_triangles(&context, &complex);
    self.render_edges(&context, &complex);
    self.render_vertices(&context);
  }

  /// Render epsilon distance bubbles around each point
  fn render_epsilon_bubbles(&self, context: &CanvasRenderingContext2d) {
    context.set_stroke_style(&JsValue::from_str("#e5e7eb")); // Light gray
    context.set_line_width(1.0);
    context.set_fill_style(&JsValue::from_str("rgba(0, 0, 0, 0.02)")); // Very subtle fill

    for point in self.cloud.points_ref() {
      context.begin_path();
      context.arc(point.0[0], point.0[1], self.epsilon, 0.0, 2.0 * std::f64::consts::PI).unwrap();
      context.fill();
      context.stroke();
    }
  }

  /// Render vertices as sharp black circles
  fn render_vertices(&self, context: &CanvasRenderingContext2d) {
    context.set_fill_style(&JsValue::from_str("#000000")); // Pure black
    context.set_stroke_style(&JsValue::from_str("#ffffff")); // White border
    context.set_line_width(2.0);

    for point in self.cloud.points_ref() {
      context.begin_path();
      context.arc(point.0[0], point.0[1], 6.0, 0.0, 2.0 * std::f64::consts::PI).unwrap();
      context.fill();
      context.stroke();
    }
  }

  /// Render edges as clean geometric lines
  fn render_edges(&self, context: &CanvasRenderingContext2d, complex: &SimplicialComplex) {
    context.set_stroke_style(&JsValue::from_str("#3b82f6")); // Clean blue
    context.set_line_width(2.0);

    let points = self.cloud.points_ref();
    let edges = complex.elements_of_dimension(1);
    for edge in edges {
      let vertices = edge.vertices();
      if vertices.len() == 2 {
        let p1 = &points[vertices[0]];
        let p2 = &points[vertices[1]];

        context.begin_path();
        context.move_to(p1.0[0], p1.0[1]);
        context.line_to(p2.0[0], p2.0[1]);
        context.stroke();
      }
    }
  }

  /// Render triangles as subtle geometric shapes
  fn render_triangles(&self, context: &CanvasRenderingContext2d, complex: &SimplicialComplex) {
    context.set_fill_style(&JsValue::from_str("rgba(59, 130, 246, 0.1)")); // Very subtle blue
    context.set_stroke_style(&JsValue::from_str("rgba(59, 130, 246, 0.3)")); // Light blue border
    context.set_line_width(1.0);

    let points = self.cloud.points_ref();
    let triangles = complex.elements_of_dimension(2);
    for triangle in triangles {
      let vertices = triangle.vertices();
      if vertices.len() == 3 {
        let p1 = &points[vertices[0]];
        let p2 = &points[vertices[1]];
        let p3 = &points[vertices[2]];

        context.begin_path();
        context.move_to(p1.0[0], p1.0[1]);
        context.line_to(p2.0[0], p2.0[1]);
        context.line_to(p3.0[0], p3.0[1]);
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
  console_error_panic_hook::set_once();
  console::log_1(&"🦀 Vietoris-Rips Demo WASM module initialized!".into());
}
