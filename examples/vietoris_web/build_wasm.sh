#!/bin/bash

echo "🦀 Building WASM module for Vietoris-Rips Demo..."

# Build the WASM module with only the wasm feature
wasm-pack build --target web --features wasm --no-default-features

if [ $? -eq 0 ]; then
    echo "✅ WASM build successful!"
    echo "📦 Generated files in pkg/ directory"
    ls -la pkg/
else
    echo "❌ WASM build failed"
    exit 1
fi 