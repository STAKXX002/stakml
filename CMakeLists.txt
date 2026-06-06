cmake_minimum_required(VERSION 3.16)
project(StakML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# ── NEW: Find OpenMP ─────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
# ── MODIFIED: Added -ffast-math to the global release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

# ── Library ──────────────────────────────────────────────────────────────────
add_library(stakml
    src/tensor.cpp
)
target_include_directories(stakml PUBLIC include)

# ── NEW: Link OpenMP to your core library ────────────────────────────────────
target_link_libraries(stakml PUBLIC OpenMP::OpenMP_CXX)

# ── Tests ────────────────────────────────────────────────────────────────────
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor stakml)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph stakml)

add_executable(bench tests/bench.cpp)
target_link_libraries(bench stakml)

add_executable(test_opts tests/test_opts.cpp)
target_link_libraries(test_opts stakml)

add_executable(test_backward tests/test_backward.cpp)
target_link_libraries(test_backward stakml)

add_executable(test_loss tests/test_loss.cpp)
target_link_libraries(test_loss stakml)

add_executable(test_optim tests/test_optim.cpp)
target_link_libraries(test_optim stakml)

add_executable(stress_test tests/stress_test.cpp)
target_link_libraries(stress_test stakml)

add_executable(test_conv tests/test_conv.cpp)
target_link_libraries(test_conv stakml)

# ── Examples ─────────────────────────────────────────────────────────────────
add_executable(mnist_mlp examples/mnist_mlp.cpp)
target_link_libraries(mnist_mlp stakml)

add_executable(demo_visual examples/demo_visual.cpp)
target_link_libraries(demo_visual stakml)

add_executable(world_cup examples/world_cup.cpp)
target_link_libraries(world_cup stakml)

add_executable(predict_match examples/predict_match.cpp)
target_link_libraries(predict_match stakml)

add_executable(group_stage examples/group_stage.cpp)
target_link_libraries(group_stage stakml)

add_executable(full_tournament examples/full_tournament.cpp)
target_link_libraries(full_tournament stakml)

add_executable(cifar_cnn examples/cifar_cnn.cpp)
target_link_libraries(cifar_cnn stakml)

# ── Convenience target ───────────────────────────────────────────────────────
add_custom_target(run_tests
    COMMAND ./test_tensor
    DEPENDS test_tensor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running StakML tests..."
)
