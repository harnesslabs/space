# Cova Examples

This directory contains interactive examples and demonstrations of the Cova computational topology library.

## Available Examples

### 🌐 Vietoris-Rips Interactive Demo

**Location**: `examples/vietoris_web/`

An interactive web-based demonstration of Vietoris-Rips complexes that lets you:
- Click to add points to a 2D plane
- Right-click to remove points
- Adjust the distance threshold (epsilon) with a slider
- Watch simplicial complexes form in real-time

**Features:**
- Real-time visualization of vertices, edges, and triangles
- Live statistics showing complex properties
- Educational tool for understanding topological data analysis

**How to run:**
```bash
cd examples/vietoris_web
cargo run
```
Then open your browser to `http://localhost:3030`

**Technologies used:**
- Rust backend with `cova-space` for topology computation
- WebAssembly for browser integration
- HTML5 Canvas for visualization
- Built-in web server using `warp`

## Adding New Examples

To add a new example:

1. Create a new directory under `examples/`
2. Add a standalone `Cargo.toml` with path dependencies to the workspace crates:
   ```toml
   [dependencies]
   cova = { path = "../../cova" }
   ```
3. Implement your example in `src/main.rs`
4. Update this README with documentation

## Requirements

- Rust 2021 edition or later
- For web examples: Modern browser with WebAssembly support

## Educational Value

These examples are designed to:
- Demonstrate practical applications of computational topology
- Provide interactive learning experiences
- Show integration patterns for the Cova library
- Serve as starting points for your own projects

Happy exploring! 🎭✨
