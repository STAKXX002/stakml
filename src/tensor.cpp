#include "stakml/tensor.hpp"
// Most of Tensor is implemented inline in tensor.hpp (it's a template-heavy,
// performance-critical class — keeping it in the header lets the compiler
// inline and optimize across translation units).
//
// This file exists so CMake has a .cpp to compile into the library object.
// Heavy non-template implementations (e.g. BLAS calls) will go here in
// later weeks when we add them.
