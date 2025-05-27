#!/bin/bash

set -e

echo "🚀 Building maximum performance WASM for Vietoris-Rips demo..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/

# Set RUSTFLAGS to enable bulk memory operations from the start
export RUSTFLAGS="-C target-feature=+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint,+simd128"

# Build with wasm-pack using release profile focused on speed
echo "📦 Building with wasm-pack (speed optimized)..."
wasm-pack build \
    --target web \
    --release \
    --out-dir pkg

# First, run wasm-opt to validate and optimize with bulk memory support
echo "⚡ Running initial optimization with bulk memory support..."
wasm-opt pkg/vietoris_web_demo_bg.wasm \
    --enable-bulk-memory \
    --enable-sign-ext \
    --enable-mutable-globals \
    --enable-nontrapping-float-to-int \
    --enable-simd \
    -O2 \
    -o pkg/vietoris_web_demo_bg.wasm

# Now run the aggressive performance optimizations
echo "🏎️  Applying maximum performance optimizations..."
wasm-opt pkg/vietoris_web_demo_bg.wasm \
    -O4 \
    --enable-bulk-memory \
    --enable-sign-ext \
    --enable-mutable-globals \
    --enable-nontrapping-float-to-int \
    --enable-simd \
    --enable-threads \
    --fast-math \
    --inline-functions-with-loops \
    --optimize-level=4 \
    --shrink-level=0 \
    --converge \
    --always-inline-max-function-size=500 \
    --flexible-inline-max-function-size=1000 \
    --partial-inlining-ifs=4 \
    -o pkg/vietoris_web_demo_bg.wasm

# Show file sizes (just for reference, not optimizing for this)
echo "📊 Build results:"
echo "Final binary size: $(du -h pkg/vietoris_web_demo_bg.wasm | cut -f1)"

echo "✅ Maximum performance build complete!"
echo "🏎️  Optimized for speed, not size - expect larger but faster WASM!" 