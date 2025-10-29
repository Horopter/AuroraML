#!/bin/bash

# CxML Build Script

echo "Building CxML..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j4

echo "Build complete!"
echo "Python module should be available as: build/cxml.cpython-*.so"
