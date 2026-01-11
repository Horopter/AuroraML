#!/bin/bash

# IngenuityML Build Script

echo "Building IngenuityML..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j4

echo "Build complete!"
echo "Python module should be available as: build/ingenuityml.cpython-*.so"
