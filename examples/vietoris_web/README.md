# Interactive Vietoris-Rips Complex Demo

An interactive web-based demonstration of Vietoris-Rips complexes built with Rust and WebAssembly.

## What is a Vietoris-Rips Complex?

A Vietoris-Rips complex is a fundamental tool in topological data analysis. Given a set of points and a distance threshold (epsilon), it constructs a simplicial complex where:

- **0-simplices (vertices)**: All the input points
- **1-simplices (edges)**: Connect points that are within distance ε of each other  
- **2-simplices (triangles)**: Form when three points are all pairwise within distance ε
- **Higher simplices**: Continue this pattern for more dimensions

This creates a geometric approximation of the underlying shape of your data!

## Features

🎮 **Interactive Visualization**
- Click to add points anywhere on the canvas
- Right-click to remove points near your cursor
- Real-time visualization of the evolving complex

📊 **Dynamic Controls**  
- Epsilon slider to adjust the distance threshold
- Live statistics showing vertex/edge/triangle counts
- Color-coded visualization (blue vertices, green edges, red triangles)

🔬 **Educational Value**
- See how topology emerges from geometry
- Understand the relationship between distance and connectivity
- Explore fundamental concepts in computational topology

## Prerequisites

1. **Rust toolchain** (install from [rustup.rs](https://rustup.rs/))
2. **wasm-pack** for WebAssembly compilation:
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

## Building and Running

1. **Build the demo:**
   ```bash
   ./build_demo.sh
   ```

2. **Start the web server:**
   ```bash
   ./serve.py
   ```

3. **Open your browser** to `http://localhost:8000`

## How to Use

1. **Add Points**: Left-click anywhere on the white canvas to place points
2. **Remove Points**: Right-click near existing points to remove them
3. **Adjust Threshold**: Drag the epsilon (ε) slider to change the distance threshold
4. **Observe**: Watch how edges and triangles appear/disappear as you change epsilon!

## Understanding the Visualization

- **Blue circles**: Vertices (0-simplices) - your input points
- **Green lines**: Edges (1-simplices) - connect points within distance ε
- **Red triangles**: Triangles (2-simplices) - form when three points are all mutually connected

The statistics panel shows live counts of each type of simplex in your complex.

## Educational Experiments to Try

1. **Create a Triangle**: Place 3 points close together, then slowly increase epsilon until a triangle forms

2. **Explore Connectivity**: Place points in two separate clusters and observe how they connect as epsilon increases

3. **Circle Approximation**: Place many points in a rough circle pattern and see how the complex approximates the circular topology

4. **Noise vs. Signal**: Add random points to a structured pattern and observe how it affects the topology

## Technical Details

- Built with `cova-space` - a Rust library for computational topology
- Uses WebAssembly for high-performance computation in the browser
- Real-time complex construction and visualization using HTML5 Canvas
- Distance calculations use squared Euclidean distance for efficiency

## Architecture

```
┌─────────────────┐    WASM     ┌──────────────────┐
│   JavaScript    │ ←--------→  │   Rust Engine    │
│   (UI & Canvas) │             │  (cova-space)    │
└─────────────────┘             └──────────────────┘
        ↓                               ↓
┌─────────────────┐             ┌──────────────────┐
│  User Actions   │             │ Vietoris-Rips    │
│ (clicks, slider)│             │    Complex       │
└─────────────────┘             └──────────────────┘
```

The demo showcases the power of combining Rust's computational efficiency with JavaScript's web capabilities for interactive mathematical visualization.

## Troubleshooting

**Build Issues:**
- Ensure `wasm-pack` is installed and in your PATH
- Check that you're in the `examples/` directory when running scripts

**Browser Issues:**
- Use a modern browser with WebAssembly support
- Ensure the local server is running (`./serve.py`)
- Check browser console for any JavaScript errors

**Performance:**
- The demo handles hundreds of points well
- For very large point sets, performance may degrade due to O(n²) distance calculations

## Extensions

Want to extend this demo? Consider adding:
- Homology computation and Betti number display
- 3D visualization for higher-dimensional data
- Animation of epsilon changes (persistence diagrams)
- Export functionality for the generated complexes
- Different distance metrics (Manhattan, etc.)

Happy exploring! 🎭✨ 