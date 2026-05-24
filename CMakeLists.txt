cmake_minimum_required(VERSION 3.16)
project(StakML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ── Library ──────────────────────────────────────────────────────────────────
add_library(stakml
    src/tensor.cpp
)
target_include_directories(stakml PUBLIC include)

# ── Tests ────────────────────────────────────────────────────────────────────
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor stakml)

# ── Examples ─────────────────────────────────────────────────────────────────
# add_executable(mnist_mlp examples/mnist_mlp.cpp)
# target_link_libraries(mnist_mlp stakml)
# (uncomment in week 4)

# ── Convenience target ───────────────────────────────────────────────────────
add_custom_target(run_tests
    COMMAND ./test_tensor
    DEPENDS test_tensor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running StakML tests..."
)
