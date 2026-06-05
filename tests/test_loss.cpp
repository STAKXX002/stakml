#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/loss.hpp"
#include "stakml/nn.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

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

// ── grad check helper (same as test_backward.cpp) ─────────────────────────────
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
        p[i] = orig + eps; float lp = loss_fn();
        p[i] = orig - eps; float lm = loss_fn();
        p[i] = orig;
        float numerical = (lp - lm) / (2.0f * eps);
        float analytic  = t->grad_->raw_ptr()[i];
        if (std::abs(numerical - analytic) > tol)
            throw std::runtime_error(
                "Grad check failed at index " + std::to_string(i)
                + ": numerical=" + std::to_string(numerical)
                + " analytic="  + std::to_string(analytic));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// log_softmax tests
// ─────────────────────────────────────────────────────────────────────────────

// Known values: log_softmax([0, 1, 2])
// softmax([0,1,2]) = [0.0900, 0.2447, 0.6652]
// log of those    = [-2.4076, -1.4076, -0.4076]  (approx)
void test_log_softmax_known_values() {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<float>{0.f, 1.f, 2.f});

    Tensor ls = ops::log_softmax(x);

    ASSERT_NEAR(ls.at({0, 0}), -2.4076f, 1e-3f);
    ASSERT_NEAR(ls.at({0, 1}), -1.4076f, 1e-3f);
    ASSERT_NEAR(ls.at({0, 2}), -0.4076f, 1e-3f);
}

// exp(log_softmax(x)) must equal softmax(x) — outputs are valid log-probs
void test_log_softmax_exp_is_softmax() {
    auto x = std::make_shared<Tensor>(Tensor::randn({4, 5}));
    Tensor ls = ops::log_softmax(x);
    Tensor sm = x->softmax();

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 5; ++j)
            ASSERT_NEAR(std::exp(ls.at({i,j})), sm.at({i,j}), 1e-5f);
}

// Each row of log_softmax should log-sum-exp to 0 (i.e., probs sum to 1)
void test_log_softmax_rows_sum_to_one_in_prob_space() {
    auto x = std::make_shared<Tensor>(Tensor::randn({8, 10}));
    Tensor ls = ops::log_softmax(x);
    for (size_t i = 0; i < 8; ++i) {
        float sum = 0.f;
        for (size_t j = 0; j < 10; ++j)
            sum += std::exp(ls.at({i,j}));
        ASSERT_NEAR(sum, 1.0f, 1e-5f);
    }
}

// Numerical stability: large logits should not produce inf/nan
void test_log_softmax_large_logits_stable() {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<float>{1000.f, 1001.f, 1002.f});
    Tensor ls = ops::log_softmax(x);
    for (size_t j = 0; j < 3; ++j) {
        ASSERT(!std::isinf(ls.at({0,j})));
        ASSERT(!std::isnan(ls.at({0,j})));
    }
    float sum = 0.f;
    for (size_t j = 0; j < 3; ++j) sum += std::exp(ls.at({0,j}));
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

// Graph node is stamped correctly
void test_log_softmax_graph_tag() {
    auto x = std::make_shared<Tensor>(Tensor::randn({2, 4}));
    Tensor ls = ops::log_softmax(x);
    ASSERT(ls.op_name_ == "log_softmax");
    ASSERT(ls.inputs_.size() == 1);
    ASSERT(ls.inputs_[0] == x);
}

// ── log_softmax backward: grad check ─────────────────────────────────────────
// loss = sum(log_softmax(x))  — not a real loss but exercises all output grads
void test_log_softmax_backward_gradcheck() {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{3, 4},
        std::vector<float>{
            0.5f, -0.3f, 1.2f, -0.7f,
            -1.0f, 2.0f, 0.1f,  0.4f,
            0.8f,  0.2f,-0.5f,  1.1f});

    auto loss_fn = [&]() -> float {
        Tensor ls = ops::log_softmax(x);
        return ls.sum_all();
    };

    // Run forward+backward once for analytic grads
    Tensor ls = ops::log_softmax(x);
    ls.backward();

    grad_check(x, loss_fn);
}

// ─────────────────────────────────────────────────────────────────────────────
// nll_loss tests
// ─────────────────────────────────────────────────────────────────────────────

// Manual: batch=1, 3 classes, label=2
// log_probs = log([0.1, 0.3, 0.6]) = [-2.303, -1.204, -0.511]
// nll = -log_probs[label=2] = 0.511
void test_nll_loss_manual_single() {
    // Pre-compute log probs
    auto lp = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<float>{
            std::log(0.1f), std::log(0.3f), std::log(0.6f)});

    std::vector<int> labels = {2};
    float loss = ops::nll_loss(*lp, labels);
    ASSERT_NEAR(loss, -std::log(0.6f), 1e-4f);
}

// Manual: batch=2
// row0 label=0: -log(0.7) = 0.357
// row1 label=1: -log(0.4) = 0.916
// mean = 0.636
void test_nll_loss_manual_batch() {
    auto lp = std::make_shared<Tensor>(
        std::vector<size_t>{2, 2},
        std::vector<float>{
            std::log(0.7f), std::log(0.3f),
            std::log(0.6f), std::log(0.4f)});

    std::vector<int> labels = {0, 1};
    float loss = ops::nll_loss(*lp, labels);
    float expected = (-std::log(0.7f) - std::log(0.4f)) / 2.0f;
    ASSERT_NEAR(loss, expected, 1e-4f);
}

// Perfect prediction should give very low loss
void test_nll_loss_perfect_prediction() {
    // log_softmax of very confident logits
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<float>{100.f, -100.f, -100.f});
    Tensor ls = ops::log_softmax(x);
    auto lp = std::make_shared<Tensor>(ls);

    float loss = ops::nll_loss(*lp, {0});  // correct class is 0
    ASSERT(loss < 1e-4f);   // should be near 0
}

// nll_loss seeds correct gradient into log_probs
// dL/d(log_probs[i][j]) = -1/batch  if j==labels[i], else 0
void test_nll_loss_grad_seed() {
    size_t batch = 3, C = 4;
    auto lp = std::make_shared<Tensor>(
        std::vector<size_t>{batch, C},
        std::vector<float>(batch*C, 0.0f));

    std::vector<int> labels = {1, 3, 0};
    ops::nll_loss(*lp, labels);

    float* g = lp->grad_->raw_ptr();
    float inv_b = -1.0f / (float)batch;
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < C; ++j) {
            float expected = ((int)j == labels[i]) ? inv_b : 0.0f;
            ASSERT_NEAR(g[i*C+j], expected, 1e-6f);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end: log_softmax + nll_loss backward through Linear
// ─────────────────────────────────────────────────────────────────────────────

// Grad check on input x through the full loss pipeline
void test_full_loss_backward_gradcheck() {
    nn::Linear fc(4, 3);
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{2, 4},
        std::vector<float>{
             0.5f, -0.3f, 1.2f, -0.7f,
            -1.0f,  2.0f, 0.1f,  0.4f});

    std::vector<int> labels = {2, 0};

    auto loss_fn = [&]() -> float {
        Tensor logits  = fc.forward(x);
        auto   lgt_ptr = std::make_shared<Tensor>(logits);
        Tensor lp      = ops::log_softmax(lgt_ptr);
        auto   lp_ptr  = std::make_shared<Tensor>(lp);
        return ops::nll_loss(*lp_ptr, labels);
    };

    // Run one forward+backward for analytic grads on W
    {
        Tensor logits  = fc.forward(x);
        auto   lgt_ptr = std::make_shared<Tensor>(logits);
        Tensor lp      = ops::log_softmax(lgt_ptr);
        auto   lp_ptr  = std::make_shared<Tensor>(lp);
        ops::nll_loss(*lp_ptr, labels);
        lp_ptr->backward();
    }

    grad_check(fc.W, loss_fn, 1e-3f, 1e-2f);
    grad_check(fc.b, loss_fn, 1e-3f, 1e-2f);
}

// Loss decreases when we take one gradient step
void test_loss_decreases_after_step() {
    nn::Linear fc(4, 3);
    auto x = std::make_shared<Tensor>(Tensor::randn({8, 4}));
    std::vector<int> labels = {0,1,2,0,1,2,0,1};
    float lr = 0.1f;

    auto compute_loss = [&]() -> float {
        Tensor logits  = fc.forward(x);
        auto   lgt_ptr = std::make_shared<Tensor>(logits);
        Tensor lp      = ops::log_softmax(lgt_ptr);
        auto   lp_ptr  = std::make_shared<Tensor>(lp);
        return ops::nll_loss(*lp_ptr, labels);
    };

    // Before step
    float loss0;
    {
        Tensor logits  = fc.forward(x);
        auto   lgt_ptr = std::make_shared<Tensor>(logits);
        Tensor lp      = ops::log_softmax(lgt_ptr);
        auto   lp_ptr  = std::make_shared<Tensor>(lp);
        loss0 = ops::nll_loss(*lp_ptr, labels);
        lp_ptr->backward();
    }

    // SGD step
    for (auto& p : fc.parameters()) {
        float* w = p->raw_ptr();
        float* g = p->grad_->raw_ptr();
        for (size_t i = 0; i < p->num_elements(); ++i)
            w[i] -= lr * g[i];
        p->zero_grad();
    }

    float loss1 = compute_loss();
    ASSERT(loss1 < loss0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Loss Function Tests\n";
    std::cout << "══════════════════════════════════════════\n";

    std::cout << "\n── log_softmax ───────────────────────────\n";
    RUN_TEST(test_log_softmax_known_values);
    RUN_TEST(test_log_softmax_exp_is_softmax);
    RUN_TEST(test_log_softmax_rows_sum_to_one_in_prob_space);
    RUN_TEST(test_log_softmax_large_logits_stable);
    RUN_TEST(test_log_softmax_graph_tag);
    RUN_TEST(test_log_softmax_backward_gradcheck);

    std::cout << "\n── nll_loss ──────────────────────────────\n";
    RUN_TEST(test_nll_loss_manual_single);
    RUN_TEST(test_nll_loss_manual_batch);
    RUN_TEST(test_nll_loss_perfect_prediction);
    RUN_TEST(test_nll_loss_grad_seed);

    std::cout << "\n── end-to-end ────────────────────────────\n";
    RUN_TEST(test_full_loss_backward_gradcheck);
    RUN_TEST(test_loss_decreases_after_step);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << pass_count << " / "
              << (pass_count + fail_count) << " passed\n";
    std::cout << "══════════════════════════════════════════\n";
    return fail_count > 0 ? 1 : 0;
}