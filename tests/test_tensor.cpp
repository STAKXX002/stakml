#include "stakml/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace stakml;

// ─────────────────────────────────────────────────────────────────────────────
// Tiny test framework
// ─────────────────────────────────────────────────────────────────────────────
static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    void name(); \
    struct _Reg_##name { _Reg_##name() { \
        ++tests_run; \
        try { name(); ++tests_passed; std::cout << "  [PASS] " #name "\n"; } \
        catch (std::exception& e) { std::cout << "  [FAIL] " #name " — " << e.what() << "\n"; } \
    }} _reg_##name; \
    void name()

#define EXPECT_EQ(a, b) \
    if (!((a) == (b))) throw std::runtime_error(std::string("Expected equal: ") \
        + std::to_string(a) + " != " + std::to_string(b))

#define EXPECT_NEAR(a, b, eps) \
    if (std::abs((float)(a) - (float)(b)) > (eps)) \
        throw std::runtime_error(std::string("Expected near: ") \
            + std::to_string(a) + " vs " + std::to_string(b))

#define EXPECT_THROW(expr) \
    { bool threw = false; try { expr; } catch (...) { threw = true; } \
      if (!threw) throw std::runtime_error("Expected exception but none thrown"); }

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(test_construction_zeros) {
    Tensor t({2, 3});
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(1), 3u);
    EXPECT_EQ(t.num_elements(), 6u);
    // all zeros
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(t.at({i,j}), 0.0f, 1e-6f);
}

TEST(test_construction_from_data) {
    Tensor t({2, 2}, {1.f, 2.f, 3.f, 4.f});
    EXPECT_NEAR(t.at({0,0}), 1.f, 1e-6f);
    EXPECT_NEAR(t.at({0,1}), 2.f, 1e-6f);
    EXPECT_NEAR(t.at({1,0}), 3.f, 1e-6f);
    EXPECT_NEAR(t.at({1,1}), 4.f, 1e-6f);
}

TEST(test_strides_row_major) {
    // shape {2,3,4} → strides must be {12, 4, 1}
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.strides_[0], 12u);
    EXPECT_EQ(t.strides_[1],  4u);
    EXPECT_EQ(t.strides_[2],  1u);
}

TEST(test_reshape_zero_copy) {
    Tensor t({2, 3}, {1,2,3,4,5,6});
    Tensor r = t.reshape({3, 2});
    // Same underlying data pointer (shared buffer — zero copy)
    if (t.data_.get() != r.data_.get())
        throw std::runtime_error("reshape should share data buffer (zero copy)");
    EXPECT_EQ(r.size(0), 3u);
    EXPECT_EQ(r.size(1), 2u);
    EXPECT_NEAR(r.at({0,0}), 1.f, 1e-6f);
    EXPECT_NEAR(r.at({1,1}), 4.f, 1e-6f);
}

TEST(test_reshape_bad_size) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({4, 2}));
}

TEST(test_transpose) {
    // [[1,2,3],[4,5,6]]  →  [[1,4],[2,5],[3,6]]
    Tensor t({2, 3}, {1,2,3,4,5,6});
    Tensor tr = t.transpose();
    EXPECT_EQ(tr.size(0), 3u);
    EXPECT_EQ(tr.size(1), 2u);
    EXPECT_NEAR(tr.at({0,0}), 1.f, 1e-6f);
    EXPECT_NEAR(tr.at({0,1}), 4.f, 1e-6f);
    EXPECT_NEAR(tr.at({1,0}), 2.f, 1e-6f);
    EXPECT_NEAR(tr.at({2,1}), 6.f, 1e-6f);
    // transposed is NOT contiguous
    EXPECT_EQ(tr.is_contiguous(), false);
    // contiguous() copy should be
    Tensor tc = tr.contiguous();
    EXPECT_EQ(tc.is_contiguous(), true);
}

TEST(test_add) {
    Tensor a({2,2}, {1,2,3,4});
    Tensor b({2,2}, {10,20,30,40});
    Tensor c = a + b;
    EXPECT_NEAR(c.at({0,0}), 11.f, 1e-6f);
    EXPECT_NEAR(c.at({1,1}), 44.f, 1e-6f);
}

TEST(test_scalar_mul) {
    Tensor a({1,3}, {1,2,3});
    Tensor b = a * 3.0f;
    EXPECT_NEAR(b.at({0,1}), 6.f, 1e-6f);
    Tensor c = 2.0f * a;
    EXPECT_NEAR(c.at({0,2}), 6.f, 1e-6f);
}

TEST(test_matmul) {
    // (2×3) @ (3×2) → (2×2)
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
    // row0: 1*7+2*9+3*11=58,  1*8+2*10+3*12=64
    // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({3,2}, {7,8,9,10,11,12});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.size(0), 2u);
    EXPECT_EQ(c.size(1), 2u);
    EXPECT_NEAR(c.at({0,0}),  58.f, 1e-4f);
    EXPECT_NEAR(c.at({0,1}),  64.f, 1e-4f);
    EXPECT_NEAR(c.at({1,0}), 139.f, 1e-4f);
    EXPECT_NEAR(c.at({1,1}), 154.f, 1e-4f);
}

TEST(test_matmul_shape_mismatch) {
    Tensor a({2,3}), b({4,2});
    EXPECT_THROW(a.matmul(b));
}

TEST(test_relu) {
    Tensor t({1,5}, {-2, -0.5f, 0, 1, 3});
    Tensor r = t.relu();
    EXPECT_NEAR(r.at({0,0}), 0.f, 1e-6f);
    EXPECT_NEAR(r.at({0,1}), 0.f, 1e-6f);
    EXPECT_NEAR(r.at({0,2}), 0.f, 1e-6f);
    EXPECT_NEAR(r.at({0,3}), 1.f, 1e-6f);
    EXPECT_NEAR(r.at({0,4}), 3.f, 1e-6f);
}

TEST(test_sigmoid) {
    Tensor t({1,3}, {0.f, 100.f, -100.f});
    Tensor s = t.sigmoid();
    EXPECT_NEAR(s.at({0,0}), 0.5f, 1e-5f);
    EXPECT_NEAR(s.at({0,1}), 1.0f, 1e-4f);
    EXPECT_NEAR(s.at({0,2}), 0.0f, 1e-4f);
}

TEST(test_softmax) {
    Tensor t({1,3}, {1.f, 2.f, 3.f});
    Tensor s = t.softmax();
    // probabilities must sum to 1
    float total = s.at({0,0}) + s.at({0,1}) + s.at({0,2});
    EXPECT_NEAR(total, 1.0f, 1e-5f);
    // largest input → largest prob
    assert(s.at({0,2}) > s.at({0,1}) && s.at({0,1}) > s.at({0,0}));
}

TEST(test_sum_all) {
    Tensor t({2,3}, {1,2,3,4,5,6});
    EXPECT_NEAR(t.sum_all(), 21.f, 1e-5f);
}

TEST(test_sum_axis) {
    // [[1,2,3],[4,5,6]] sum axis=0 → [5,7,9]
    Tensor t({2,3}, {1,2,3,4,5,6});
    Tensor s = t.sum(0);
    EXPECT_EQ(s.num_elements(), 3u);
    EXPECT_NEAR(s.at({0}), 5.f, 1e-5f);
    EXPECT_NEAR(s.at({1}), 7.f, 1e-5f);
    EXPECT_NEAR(s.at({2}), 9.f, 1e-5f);
}

TEST(test_xavier_shape) {
    Tensor w = Tensor::xavier({64, 32});
    EXPECT_EQ(w.size(0), 64u);
    EXPECT_EQ(w.size(1), 32u);
    // mean should be near 0 for large tensor
    EXPECT_NEAR(w.mean_all(), 0.0f, 0.1f);
}

TEST(test_shared_memory_reshape) {
    // Modifying reshaped tensor must modify original (shared buffer)
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b = a.reshape({6});
    b.raw_ptr()[0] = 99.f;
    EXPECT_NEAR(a.at({0,0}), 99.f, 1e-6f); // shared!
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n══════════════════════════════════\n";
    std::cout << "  StakML — Tensor Tests\n";
    std::cout << "══════════════════════════════════\n\n";

    // Visual demo
    std::cout << "── Demo ──────────────────────────\n";
    Tensor w = Tensor::xavier({3, 3});
    w.print("xavier(3×3)");

    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({3,2}, {7,8,9,10,11,12});
    a.matmul(b).print("matmul result");

    a.relu().print("relu of a");

    std::cout << "\n── Tests ─────────────────────────\n";
    // (tests self-register and run via static constructors above)

    std::cout << "\n══════════════════════════════════\n";
    std::cout << "  " << tests_passed << " / " << tests_run << " passed\n";
    std::cout << "══════════════════════════════════\n\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
