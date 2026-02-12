#!/bin/bash
# Script to recompile RTDLite library after fixing C++ code

cd "$(dirname "$0")"

echo "Cleaning previous build..."
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -type f -delete
find . -name "*.dylib" -type f -delete
find . -name "*.dll" -type f -delete

echo "Recompiling RTDLite library..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Library should be in: RTDLite/rtd_lite.so (or .dylib/.dll)"
else
    echo "✗ Compilation failed!"
    exit 1
fi

