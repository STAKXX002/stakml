#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include <iostream>
#include <cmath>
#include <string>

using namespace stakml;

static int pass_count = 0;
static int fail_count = 0;

#define RUN_TEST(fn) do {                               \
    try {                                               \
        fn();                                           \
        std::cout << "  [PASS] " #fn "\n";              \
        ++pass_count;                                   \
    } catch (std::exception& e) {                       \
        std::cout << "  [FAIL] " #fn "\n"               \
                  << "        " << e.what() << "\n";    \
        ++fail_count;                                   \
    }                                                   \
} while(0)

#define ASSERT(cond) do {                               \
    if (!(cond)) throw std::runtime_error(              \
        "Assertion failed: " #cond                      \
        " at line " + std::to_string(__LINE__));        \
} while(0)

#define ASSERT_NEAR(a, b, eps) do {                                 \
    if (std::abs((float)(a) - (float)(b)) > (float)(eps))           \
        throw std::runtime_error(                                   \
            std::string("Expected ~") + std::to_string((float)(b))  \
            + " got " + std::to_string((float)(a))                  \
            + " at line " + std::to_string(__LINE__));              \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Numerical gradient checker
//
// For each element i of tensor t, computes:
//   (loss(t[i] + eps) - loss(t[i] - eps)) / (2*eps)
// and compares to t->grad_->raw_ptr()[i].
//
// loss_fn must return a scalar float.
// ─────────────────────────────────────────────────────────────────────────────
void grad_check(
    std::shared_ptr<Tensor> t,
    std::function<float()> loss_fn,
    float eps = 1e-3f,
    float tol = 1e-2f)
{
    size_t n = t->num_elements();
    float* p = t->raw_ptr();

    for (size_t i = 0; i < n; ++i) {
        float orig = p[i];

        p[i] = orig + eps;
        float lp = loss_fn();

        p[i] = orig - eps;
        float lm = loss_fn();

        p[i] = orig;  // restore

        float numerical = (lp - lm) / (2.0f * eps);
        float analytic  = t->grad_->raw_ptr()[i];

        if (std::abs(numerical - analytic) > tol) {
            throw std::runtime_error(
                "Grad check failed at index " + std::to_string(i)
                + ": numerical=" + std::to_string(numerical)
                + " analytic=" + std::to_string(analytic));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ── relu backward: manual check ───────────────────────────────────────────────
// y = relu([-1, 2, -3, 4])
// dy/dx = [0, 1, 0, 1]
// so x->grad after backward() = [0, 1, 0, 1]
void test_relu_backward_manual() {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 4},
        std::vector<float>{-1.f, 2.f, -3.f, 4.f});

    Tensor y = ops::relu(x);
    y.backward();

    ASSERT(x->grad_ != nullptr);
    ASSERT_NEAR(x->grad_->raw_ptr()[0], 0.0f, 1e-5f);  // -1 → dead
    ASSERT_NEAR(x->grad_->raw_ptr()[1], 1.0f, 1e-5f);  // 2  → alive
    ASSERT_NEAR(x->grad_->raw_ptr()[2], 0.0f, 1e-5f);  // -3 → dead
    ASSERT_NEAR(x->grad_->raw_ptr()[3], 1.0f, 1e-5f);  // 4  → alive
}

// ── relu backward: grad check ─────────────────────────────────────────────────
void test_relu_backward_gradcheck() {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{2, 3},
        std::vector<float>{0.5f, -1.f, 2.f, -0.3f, 1.5f, -2.f});

    // loss = sum(relu(x))
    auto loss_fn = [&]() -> float {
        Tensor y = ops::relu(x);
        return y.sum_all();
    };

    // run backward once to get analytic grads
    Tensor y = ops::relu(x);
    y.backward();

    grad_check(x, loss_fn);
}

// ── add_bias backward: manual check ───────────────────────────────────────────
// out = x + bias (broadcast)
// d_x    = d_out (same shape, pass-through)
// d_bias = sum of d_out over batch
void test_add_bias_backward_manual() {
    // x: {2,3}, bias: {1,3}
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{2, 3},
        std::vector<float>{1, 2, 3, 4, 5, 6});
    auto b = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<float>{0.1f, 0.2f, 0.3f});

    Tensor out = ops::add_bias(x, b);
    out.backward();   // seeds d_out = ones

    // d_x should be all 1s (pass-through)
    ASSERT(x->grad_ != nullptr);
    for (size_t i = 0; i < 6; ++i)
        ASSERT_NEAR(x->grad_->raw_ptr()[i], 1.0f, 1e-5f);

    // d_bias should be [2, 2, 2] (sum of two rows of ones)
    ASSERT(b->grad_ != nullptr);
    ASSERT_NEAR(b->grad_->raw_ptr()[0], 2.0f, 1e-5f);
    ASSERT_NEAR(b->grad_->raw_ptr()[1], 2.0f, 1e-5f);
    ASSERT_NEAR(b->grad_->raw_ptr()[2], 2.0f, 1e-5f);
}

// ── matmul backward: manual check ─────────────────────────────────────────────
// C = A @ B,   A:{2,3}, B:{3,2}
// dA = dC @ B.T,  dB = A.T @ dC
// with dC = ones{2,2}:
//   dA = ones{2,2} @ B.T{2,3} → each row of dA = sum of rows of B.T = col-sums of B
//   dB = A.T{3,2} @ ones{2,2} → each col of dB = sum of cols of A.T = row-sums of A
void test_matmul_backward_manual() {
    auto a = std::make_shared<Tensor>(
        std::vector<size_t>{2, 3},
        std::vector<float>{1, 2, 3,
                           4, 5, 6});
    auto b = std::make_shared<Tensor>(
        std::vector<size_t>{3, 2},
        std::vector<float>{7,  8,
                           9,  10,
                           11, 12});

    Tensor c = ops::matmul(a, b);
    c.backward();   // dC = ones{2,2}

    // dA = ones{2,2} @ B.T
    // B.T = [[7,9,11],[8,10,12]]
    // dA row0 = row1 = [7+8, 9+10, 11+12] = [15, 19, 23]
    ASSERT(a->grad_ != nullptr);
    ASSERT_NEAR(a->grad_->raw_ptr()[0], 15.f, 1e-3f);
    ASSERT_NEAR(a->grad_->raw_ptr()[1], 19.f, 1e-3f);
    ASSERT_NEAR(a->grad_->raw_ptr()[2], 23.f, 1e-3f);
    ASSERT_NEAR(a->grad_->raw_ptr()[3], 15.f, 1e-3f);
    ASSERT_NEAR(a->grad_->raw_ptr()[4], 19.f, 1e-3f);
    ASSERT_NEAR(a->grad_->raw_ptr()[5], 23.f, 1e-3f);

    // dB = A.T @ ones{2,2}
    // A.T = [[1,4],[2,5],[3,6]]
    // dB col0 = col1 = [1+4, 2+5, 3+6] = [5, 7, 9]
    ASSERT(b->grad_ != nullptr);
    ASSERT_NEAR(b->grad_->raw_ptr()[0], 5.f, 1e-3f);
    ASSERT_NEAR(b->grad_->raw_ptr()[1], 5.f, 1e-3f);
    ASSERT_NEAR(b->grad_->raw_ptr()[2], 7.f, 1e-3f);
    ASSERT_NEAR(b->grad_->raw_ptr()[3], 7.f, 1e-3f);
    ASSERT_NEAR(b->grad_->raw_ptr()[4], 9.f, 1e-3f);
    ASSERT_NEAR(b->grad_->raw_ptr()[5], 9.f, 1e-3f);
}

// ── matmul backward: grad check ───────────────────────────────────────────────
void test_matmul_backward_gradcheck() {
    auto a = std::make_shared<Tensor>(Tensor::randn({3, 4}));
    auto b = std::make_shared<Tensor>(Tensor::randn({4, 2}));

    // loss = sum(a @ b)
    auto loss_fn_a = [&]() -> float {
        return a->matmul(*b).sum_all();
    };
    auto loss_fn_b = [&]() -> float {
        return a->matmul(*b).sum_all();
    };

    Tensor c = ops::matmul(a, b);
    c.backward();

    grad_check(a, loss_fn_a);
    grad_check(b, loss_fn_b);
}

// ── Linear backward: grad check on W and b ────────────────────────────────────
// This is the real integration test — backward through add_bias(matmul(x,W), b)
void test_linear_backward_gradcheck() {
    nn::Linear fc(4, 3);
    auto x = std::make_shared<Tensor>(Tensor::randn({2, 4}));

    // loss = sum(fc(x))
    auto loss_fn_W = [&]() -> float {
        Tensor out = fc.forward(x);
        return out.sum_all();
    };
    auto loss_fn_b = [&]() -> float {
        Tensor out = fc.forward(x);
        return out.sum_all();
    };

    // run one forward+backward to get analytic grads
    Tensor out = fc.forward(x);
    out.backward();

    grad_check(fc.W, loss_fn_W);
    grad_check(fc.b, loss_fn_b);
}

// ── backward() on a leaf doesn't crash ────────────────────────────────────────
void test_backward_on_leaf() {
    auto x = std::make_shared<Tensor>(Tensor::ones({2, 2}));
    // no backward_fn_, no inputs_ — should be a no-op, not a crash
    x->backward();
    // grad should be seeded to ones
    ASSERT(x->grad_ != nullptr);
    for (size_t i = 0; i < 4; ++i)
        ASSERT_NEAR(x->grad_->raw_ptr()[i], 1.0f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Backward / Autograd Tests\n";
    std::cout << "══════════════════════════════════════════\n";

    std::cout << "\n── relu ──────────────────────────────────\n";
    RUN_TEST(test_relu_backward_manual);
    RUN_TEST(test_relu_backward_gradcheck);

    std::cout << "\n── add_bias ──────────────────────────────\n";
    RUN_TEST(test_add_bias_backward_manual);

    std::cout << "\n── matmul ────────────────────────────────\n";
    RUN_TEST(test_matmul_backward_manual);
    RUN_TEST(test_matmul_backward_gradcheck);

    std::cout << "\n── Linear (integration) ──────────────────\n";
    RUN_TEST(test_linear_backward_gradcheck);

    std::cout << "\n── edge cases ────────────────────────────\n";
    RUN_TEST(test_backward_on_leaf);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << pass_count << " / " << (pass_count + fail_count) << " passed\n";
    std::cout << "══════════════════════════════════════════\n";
    return fail_count > 0 ? 1 : 0;
}