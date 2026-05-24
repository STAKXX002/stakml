#!/bin/bash
# StakML cleanup script
# Run this to nuke the build and start fresh: ./cleanup.sh
# Run with --all to also remove generated data files: ./cleanup.sh --all

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Cleaning StakML project at: $PROJECT_DIR"

# Remove build artifacts
if [ -d "$PROJECT_DIR/build" ]; then
    rm -rf "$PROJECT_DIR/build"
    echo "    [x] Removed build/"
fi

# Remove CMake cache files that sometimes leak to root
rm -f "$PROJECT_DIR/CMakeCache.txt"
rm -rf "$PROJECT_DIR/CMakeFiles"
rm -f "$PROJECT_DIR/cmake_install.cmake"
rm -f "$PROJECT_DIR/Makefile"
echo "    [x] Removed stray CMake files"

# Remove compiled test/example binaries if they ended up in root
find "$PROJECT_DIR" -maxdepth 1 -type f -executable ! -name "*.sh" -delete
echo "    [x] Removed stray binaries"

if [ "$1" == "--all" ]; then
    # Remove downloaded datasets (MNIST etc.)
    rm -rf "$PROJECT_DIR/data"
    echo "    [x] Removed data/"
fi

echo ""
echo "==> Done. To rebuild:"
echo "    mkdir build && cd build && cmake .. && make -j$(nproc)"
